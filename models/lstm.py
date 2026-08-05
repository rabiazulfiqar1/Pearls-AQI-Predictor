"""
Model training — multi-horizon AQI forecasting.

Reads the engineered feature group from Hopsworks and trains an LSTM
that predicts AQI 24h, 48h, and 72h ahead in a single forward pass,
using separate output heads per horizon.

Changes from the previous version (train_aqi_model.py) / Day 10
(models/train_lstm.py), consolidated into one script:

  1. GAP-AWARE LOADING. fg.read() only returns rows that actually made
     it into Hopsworks — if the hourly pipeline failed for a few hours,
     those timestamps are just MISSING, not NaN. The old code indexed
     windows by row position, so a failed run would silently stitch
     together a "72-hour window" that actually spanned more real time
     than that, with no error and no log line. This version reindexes
     onto a full hourly grid first (utils.sequences.reindex_to_hourly_grid),
     so gaps become explicit NaN rows, short gaps get interpolated, and
     build_sequences() refuses to build any window that spans a gap it
     can't interpolate away.
  2. PROPER VALIDATION SPLIT. The old script fed the test set into
     validation_data=, which means test-set loss was watched every
     epoch during training decisions — a slow leak. This version holds
     out a validation slice from the training pool only; the test set
     is untouched until the final evaluate() call.
  3. EARLY STOPPING + DROPOUT + multi-head architecture (matching
     Day 10, which is the stronger design) instead of 50 fixed epochs
     on a single shared Dense output.
  4. SEEDED for reproducible RMSE comparisons across runs.

IMPORTANT — before running:
  1. Check utils/features.py (or df.columns after the fetch below) for
     the ACTUAL target column names. TARGET_COLUMNS below is a guess —
     adjust to match your feature group.
  2. Any column that leaks future information (raw `aqi` as a lag
     feature is fine, but a target column must never end up in
     feature_cols).

Usage:
    python -m pipelines.train_aqi_model
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

tf.random.set_seed(SEED)
np.random.seed(SEED)

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

def build_model(lookback: int, n_features: int) -> keras.Model:
    inp = keras.Input(shape=(lookback, n_features), name="sequence_input")

    x = layers.LSTM(64, return_sequences=True, name="lstm_1")(inp)
    x = layers.Dropout(0.2, name="drop_1")(x)
    x = layers.LSTM(32, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(0.2, name="drop_2")(x)
    x = layers.Dense(16, activation="relu", name="shared_dense")(x)

    out_24h = layers.Dense(1, name="out_24h")(x)
    out_48h = layers.Dense(1, name="out_48h")(x)
    out_72h = layers.Dense(1, name="out_72h")(x)

    model = keras.Model(inputs=inp, outputs=[out_24h, out_48h, out_72h])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={"out_24h": "mse", "out_48h": "mse", "out_72h": "mse"},
        loss_weights={"out_24h": LOSS_WEIGHTS[0], "out_48h": LOSS_WEIGHTS[1], "out_72h": LOSS_WEIGHTS[2]},
        metrics={"out_24h": "mae", "out_48h": "mae", "out_72h": "mae"},
    )
    return model


def train(model: keras.Model, X_train, y_train, X_val, y_val) -> keras.callbacks.History:
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
        verbose=1,
    )
    return history


def evaluate_model(model: keras.Model, X_test, y_test) -> dict:
    preds = model.predict(X_test, verbose=0)  # list of 3 arrays, one per head
    results = {}
    for i, h in enumerate(HORIZONS):
        results[h] = evaluate(y_test[:, i], preds[i].squeeze())
    return results


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

    model = build_model(lookback=LOOKBACK_HOURS, n_features=X_train.shape[2])
    model.summary(print_fn=logger.info)

    history = train(model, X_train, y_train, X_val, y_val)
    epochs_run = len(history.history["loss"])
    logger.info("Stopped at epoch %d / %d", epochs_run, MAX_EPOCHS)

    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Training loss")
    plt.savefig("training_loss.png")
    logger.info("Saved loss curve to training_loss.png")

    results = evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("LSTM RESULTS (holdout, never seen during training or early stopping)")
    print("=" * 60)
    for h in HORIZONS:
        r = results[h]
        print(f"  {h}h  RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}  R²={r['r2']:.3f}")
    print("=" * 60)

    model.save("aqi_multi_horizon_lstm.keras")
    logger.info("Model saved to aqi_multi_horizon_lstm.keras")

    return results


def main() -> None:
    try:
        results = run_training()
        logger.info("Training complete. Results per horizon: %s", results)
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()