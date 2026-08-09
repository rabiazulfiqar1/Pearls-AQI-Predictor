"""
Model training — multi-horizon AQI forecasting.

Reads the engineered feature group from Hopsworks and trains an LSTM
that predicts AQI 24h, 48h, and 72h ahead in a single forward pass,
using separate output heads per horizon.

  1. GAP-AWARE LOADING. fg.read() only returns rows that actually made
     it into Hopsworks — if the hourly pipeline failed for a few hours,
     those timestamps are just MISSING, not NaN. This version reindexes
     onto a full hourly grid first (utils.sequences.reindex_to_hourly_grid),
     so gaps become explicit NaN rows, short gaps get interpolated, and
     build_sequences() refuses to build any window that spans a gap it
     can't interpolate away.
  2. EARLY STOPPING + DROPOUT + multi-head architecture instead of 50 fixed epochs
     on a single shared Dense output.
  3. SEEDED for reproducible RMSE comparisons across runs.

Usage:
    python -m models.lstm
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress TF noise

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

keras = tf.keras
layers = tf.keras.layers
regularizers = tf.keras.regularizers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger
from utils.metrics import evaluate
from utils.sequences import build_sequences, get_feature_cols, reindex_to_hourly_grid

logger = get_logger(__name__)

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

TARGET_COLUMNS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
HORIZONS = [24, 48, 72]

LOOKBACK_HOURS = 72          # hours of history the model sees per prediction
GAP_INTERPOLATE_LIMIT = 2    # gaps longer than this are left as NaN, not filled
HOLDOUT_FRAC = 0.15          # last 15% of the timeline, held fully blind
VAL_FRAC = 0.10              # slice of the *training pool* used for early stopping
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
LOSS_WEIGHTS = [1.0, 1.2, 1.5]   # 24h / 48h / 72h — heavier weight on the harder horizon
SEED = 42
RESULTS_CSV = "/kaggle/working/lstm_results.csv"

tf.random.set_seed(SEED)
np.random.seed(SEED)

# Hyperparameter / architecture search space 
# Each config fully describes one architecture variant.
# (each entry is a full train-to-convergence run under EarlyStopping).
HP_CONFIGS = [
    {
        "name": "baseline_64_32",
        "lstm_units": [64, 32],
        "dropout": 0.2,
        "recurrent_dropout": 0.0,
        "l2": 0.0,
        "dense_units": 16,
        "learning_rate": 1e-3,
        "bidirectional": False,
    },
    {
        "name": "wider_128_64",
        "lstm_units": [128, 64],
        "dropout": 0.3,
        "recurrent_dropout": 0.1,
        "l2": 1e-5,
        "dense_units": 32,
        "learning_rate": 1e-3,
        "bidirectional": False,
    },
    {
        "name": "deeper_64_64_32",
        "lstm_units": [64, 64, 32],
        "dropout": 0.25,
        "recurrent_dropout": 0.0,
        "l2": 1e-5,
        "dense_units": 16,
        "learning_rate": 5e-4,
        "bidirectional": False,
    },
    {
        "name": "bidirectional_64_32",
        "lstm_units": [64, 32],
        "dropout": 0.2,
        "recurrent_dropout": 0.0,
        "l2": 0.0,
        "dense_units": 16,
        "learning_rate": 1e-3,
        "bidirectional": True,
    },
    {
        "name": "regularized_96_48_lowlr",
        "lstm_units": [96, 48],
        "dropout": 0.3,
        "recurrent_dropout": 0.1,
        "l2": 1e-4,
        "dense_units": 24,
        "learning_rate": 3e-4,
        "bidirectional": False,
    },
]

def load_feature_group() -> pd.DataFrame:
    logger.info("Connecting to Hopsworks and reading '%s' v%d",
                FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)
    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d rows, %d columns from Hopsworks", *df.shape)
    return df

def chronological_split(X: np.ndarray, y: np.ndarray, timestamps: pd.DatetimeIndex) -> tuple:
    n = len(X)
    holdout_start = int(n * (1 - HOLDOUT_FRAC))
    val_start = int(holdout_start * (1 - VAL_FRAC))

    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:holdout_start], y[val_start:holdout_start]
    X_test, y_test = X[holdout_start:], y[holdout_start:]
    ts_test = timestamps[holdout_start:]

    logger.info("Split — train: %d  val: %d  holdout: %d",
                len(X_train), len(X_val), len(X_test))
    return X_train, y_train, X_val, y_val, X_test, y_test, ts_test


# fit on train only, applied to val/test
def scale(X_train, X_val, X_test) -> tuple:
    n_train, t, f = X_train.shape
    scaler = StandardScaler()
    X_train_2d = scaler.fit_transform(X_train.reshape(-1, f))
    X_val_2d = scaler.transform(X_val.reshape(-1, f))
    X_test_2d = scaler.transform(X_test.reshape(-1, f))
    return (
        X_train_2d.reshape(n_train, t, f),
        X_val_2d.reshape(len(X_val), t, f),
        X_test_2d.reshape(len(X_test), t, f),
        scaler,
    )

def build_model(lookback: int, n_features: int, config: dict) -> keras.Model:
    """
    Build a stacked-LSTM multi-output model from a config dict:
      lstm_units:        list[int]  — one entry per LSTM layer, stacked
      dropout:            float     — Dropout after every LSTM layer
      recurrent_dropout:  float     — recurrent_dropout inside each LSTM
      l2:                 float     — L2 weight decay on LSTM + Dense kernels (0 disables)
      dense_units:        int       — width of the shared Dense layer before output heads
      learning_rate:       float
      bidirectional:       bool     — wrap every LSTM layer in Bidirectional
    """
    reg = regularizers.l2(config["l2"]) if config["l2"] > 0 else None
    inp = keras.Input(shape=(lookback, n_features), name="sequence_input")
    x = inp
    n_layers = len(config["lstm_units"])
    for i, units in enumerate(config["lstm_units"]):
        return_sequences = i < n_layers - 1
        lstm_layer = layers.LSTM(
            units,
            return_sequences=return_sequences,
            recurrent_dropout=config["recurrent_dropout"],
            kernel_regularizer=reg,
            name=f"lstm_{i + 1}",
        )
        if config["bidirectional"]:
            lstm_layer = layers.Bidirectional(lstm_layer, name=f"bilstm_{i + 1}")
        x = lstm_layer(x)
        x = layers.Dropout(config["dropout"], name=f"drop_{i + 1}")(x)

    x = layers.Dense(config["dense_units"], activation="relu", kernel_regularizer=reg, name="shared_dense")(x)

    out_24h = layers.Dense(1, name="out_24h")(x)
    out_48h = layers.Dense(1, name="out_48h")(x)
    out_72h = layers.Dense(1, name="out_72h")(x)

    model = keras.Model(inputs=inp, outputs=[out_24h, out_48h, out_72h], name=config["name"])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss={"out_24h": "mse", "out_48h": "mse", "out_72h": "mse"},
        loss_weights={"out_24h": LOSS_WEIGHTS[0], "out_48h": LOSS_WEIGHTS[1], "out_72h": LOSS_WEIGHTS[2]},
        metrics={"out_24h": "mae", "out_48h": "mae", "out_72h": "mae"},
    )
    return model


def plot_loss_history(history: keras.callbacks.History, model_name: str) -> None:
    plot_dir = Path("results/loss_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"{model_name}_loss.png"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history.get("loss", []), label="train_loss")
    ax.plot(history.history.get("val_loss", []), label="val_loss")
    ax.set_title(f"{model_name} loss history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved loss plot to %s", plot_path)


def train(
    model: keras.Model,
    X_train,
    y_train,
    X_val,
    y_val,
    verbose: int = 1,
) -> keras.callbacks.History:
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1,
    )
    history = model.fit(
        X_train,
        [y_train[:, 0], y_train[:, 1], y_train[:, 2]],
        validation_data=(X_val, [y_val[:, 0], y_val[:, 1], y_val[:, 2]]),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=verbose,
    )
    plot_loss_history(history, getattr(model, "name", "lstm_model"))
    return history


def evaluate_model(model: keras.Model, X_test, y_test) -> dict:
    preds = model.predict(X_test, verbose=0)  # list of 3 arrays, one per head
    results = {}
    for i, h in enumerate(HORIZONS):
        results[h] = evaluate(y_test[:, i], preds[i].squeeze())
    return results

def append_results(rows: list[dict]):
    """Append rows to the shared results CSV so LSTM / classical / SHAP-pruned
    results can be joined later for champion-model selection."""
    new_df = pd.DataFrame(rows)
    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(RESULTS_CSV, index=False)
    logger.info("Wrote %d rows to %s (%d total)", len(rows), RESULTS_CSV, len(new_df))


def run_training() -> dict:
    df = load_feature_group()

    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected target columns not found: {missing}. "
            f"Check utils/features.py and update TARGET_COLUMNS at the top "
            f"of this script. Available columns: {list(df.columns)}"
        )

    # Make gaps explicit (NaN rows) instead of silently missing rows,
    # so build_sequences can refuse to build windows that span them.
    df = reindex_to_hourly_grid(df, interpolate_limit=GAP_INTERPOLATE_LIMIT)

    feature_cols = get_feature_cols(df, TARGET_COLUMNS)
    logger.info("Using %d feature columns, %d targets", len(feature_cols), len(TARGET_COLUMNS))

    X, y, timestamps = build_sequences(df, feature_cols, TARGET_COLUMNS, lookback=LOOKBACK_HOURS)

    X_train, y_train, X_val, y_val, X_test, y_test, _ = chronological_split(X, y, timestamps)
    X_train, X_val, X_test, _ = scale(X_train, X_val, X_test)

    n_features = X_train.shape[2]

    # ── Hyperparameter / architecture search using VAL only ─────────
    search_log = []
    trained_models = {}

    
    for config in HP_CONFIGS:
        logger.info("=== Training config: %s ===", config["name"])
        model = build_model(LOOKBACK_HOURS, n_features, config)
        history = train(model, X_train, y_train, X_val, y_val, verbose=0)
        epochs_run = len(history.history["loss"])
        best_val_loss = min(history.history["val_loss"])
        val_metrics = evaluate(model, X_val, y_val)

        logger.info(
            "    %s | epochs=%d | best_val_loss=%.3f | val RMSE 24h/48h/72h = %.2f/%.2f/%.2f",
            config["name"], epochs_run, best_val_loss,
            val_metrics[24]["rmse"], val_metrics[48]["rmse"], val_metrics[72]["rmse"],
        )

        search_log.append({"config": config["name"], "epochs_run": epochs_run, "best_val_loss": best_val_loss, "val_metrics": val_metrics})
        trained_models[config["name"]] = model

        for h in HORIZONS:
            append_results([{
                "model": f"LSTM_{config['name']}", "horizon": h, "split": "val",
                **val_metrics[h],
            }])


    # ── Pick the winner by best (lowest) aggregate val_loss ────────────────────
    winner_entry = min(search_log, key=lambda r: r["best_val_loss"])
    winner_name = winner_entry["config"]
    winner_model = trained_models[winner_name]
    logger.info("Winning config: %s (best_val_loss=%.3f)", winner_name, winner_entry["best_val_loss"])

    print("\n" + "=" * 70)
    print("VALIDATION COMPARISON — all architecture/hyperparameter configs")
    print("=" * 70)
    for r in search_log:
        tag = " <== SELECTED" if r["config"] == winner_name else ""
        print(f"\n{r['config']}{tag}  (epochs run: {r['epochs_run']}, best_val_loss={r['best_val_loss']:.3f})")
        for h in HORIZONS:
            m = r["val_metrics"][h]
            print(f"  {h}h  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")
    print("=" * 70)

    # ── Holdout is touched on the winning model only ─────────────
    holdout_results = evaluate(winner_model, X_test, y_test)

    print("\n" + "=" * 60)
    print(f"LSTM HOLDOUT RESULTS — winning config: {winner_name}")
    print("(holdout, never seen during training, val-based selection, or early stopping)")
    print("=" * 60)
    for h in HORIZONS:
        r = holdout_results[h]
        print(f"  {h}h  RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}  R2={r['r2']:.3f}")
    print("=" * 60)

    for h in HORIZONS:
        append_results([{
            "model": f"LSTM_{winner_name}_WINNER", "horizon": h, "split": "holdout",
            **holdout_results[h],
        }])

    winner_model.save(f"/kaggle/working/aqi_multi_horizon_lstm_{winner_name}.keras")
    logger.info("Winning model saved to /kaggle/working/aqi_multi_horizon_lstm_%s.keras", winner_name)
    logger.info("All val + holdout results appended to %s for cross-model comparison", RESULTS_CSV)


def main() -> None:
    try:
        run_training()
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()