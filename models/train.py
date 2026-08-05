"""
Training pipeline — daily model (re)training.

Flow:
    1. Fetch the full historical feature group from Hopsworks (same
       one backfill_pipeline.py / feature_pipeline.py maintain).
    2. Time-based train/test split (no shuffling — this is a time
       series, shuffling would leak future info into training).
    3. Train + evaluate a few candidate models on the SAME split:
         - Ridge Regression (linear baseline)
         - Random Forest (non-linear baseline)
         - A small feedforward NN (TensorFlow/Keras)
       All three are trained as direct multi-output regressors —
       one model predicts all of [aqi_t+24, aqi_t+48, aqi_t+72] at
       once, no separate model per horizon needed for Ridge/RF, and
       a 3-unit output layer for the NN.
    4. Score each with RMSE, MAE, R² per horizon, pick the model with
       the best average RMSE across horizons.
    5. Register the winner in the Hopsworks Model Registry.

Run daily (e.g. via the same GitHub Actions setup as feature_pipeline.py,
just on a daily cron instead of hourly).

Usage:
    python -m pipelines.training_pipeline
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aqi_multi_horizon_model"

TARGET_COLUMNS = ["aqi_target_24h", "aqi_target_48h", "aqi_target_72h"]
NON_FEATURE_COLUMNS = ["timestamp"] + TARGET_COLUMNS

TEST_FRACTION = 0.15
LOCAL_MODEL_DIR = Path("model_artifacts")


def load_feature_group() -> pd.DataFrame:
    logger.info("Connecting to Hopsworks and reading '%s' v%d",
                FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)
    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=TARGET_COLUMNS).reset_index(drop=True)
    logger.info("Loaded %d rows, %d columns after dropping unlabeled tail",
                *df.shape)
    return df


def time_split(df: pd.DataFrame, feature_cols: list[str]):
    split_idx = int(len(df) * (1 - TEST_FRACTION))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    logger.info("Train rows: %d, Test rows: %d", len(train_df), len(test_df))
    return (
        train_df[feature_cols].values, train_df[TARGET_COLUMNS].values,
        test_df[feature_cols].values, test_df[TARGET_COLUMNS].values,
    )


def build_nn(n_features: int, n_targets: int):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(n_features,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_targets),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-horizon RMSE/MAE/R², plus the across-horizon average RMSE
    used to compare candidate models."""
    per_horizon = {}
    for i, col in enumerate(TARGET_COLUMNS):
        per_horizon[col] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
        }
    avg_rmse = float(np.mean([m["rmse"] for m in per_horizon.values()]))
    return {"per_horizon": per_horizon, "avg_rmse": avg_rmse}


def train_candidates(X_train, y_train, X_test, y_test, n_features, n_targets):
    """Train each candidate, return {name: (fitted_model, metrics, kind)}."""
    candidates = {}

    logger.info("Training Ridge Regression")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    candidates["ridge"] = (ridge, evaluate(y_test, ridge.predict(X_test)), "sklearn")

    logger.info("Training Random Forest")
    rf = RandomForestRegressor(n_estimators=200, max_depth=12,
                                n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    candidates["random_forest"] = (rf, evaluate(y_test, rf.predict(X_test)), "sklearn")

    logger.info("Training feedforward NN")
    nn = build_nn(n_features, n_targets)
    nn.fit(X_train, y_train, validation_data=(X_test, y_test),
           epochs=50, batch_size=32, verbose=0)
    candidates["neural_net"] = (nn, evaluate(y_test, nn.predict(X_test)), "keras")

    for name, (_, metrics, _) in candidates.items():
        logger.info("%s -> avg RMSE: %.3f | per-horizon: %s",
                    name, metrics["avg_rmse"], metrics["per_horizon"])
    return candidates


def save_locally(name: str, model, kind: str) -> Path:
    LOCAL_MODEL_DIR.mkdir(exist_ok=True)
    if kind == "sklearn":
        path = LOCAL_MODEL_DIR / f"{name}.joblib"
        joblib.dump(model, path)
    else:
        path = LOCAL_MODEL_DIR / f"{name}.keras"
        model.save(path)
    logger.info("Saved winning model locally to %s", path)
    return path


def register_in_model_registry(project, name: str, model_path: Path, metrics: dict):
    """
    Register the winning model in the Hopsworks Model Registry.

    NOTE: this assumes `get_feature_store()` in utils/hopsworks_client.py
    logs in via `hopsworks.login()` under the hood. If that module
    already exposes the `project` object (or a `get_project()` helper),
    use that instead of re-deriving it here — I don't have visibility
    into that file's exact contents from this conversation.
    """
    mr = project.get_model_registry()
    flat_metrics = {
        f"{horizon}_{metric}": val
        for horizon, scores in metrics["per_horizon"].items()
        for metric, val in scores.items()
    }
    flat_metrics["avg_rmse"] = metrics["avg_rmse"]

    model = mr.python.create_model(
        name=MODEL_NAME,
        metrics=flat_metrics,
        description=f"Best-of-3 candidate ({name}) for 24/48/72h AQI forecasting",
    )
    model.save(str(model_path.parent))
    logger.info("Registered model '%s' (winner: %s) in Model Registry", MODEL_NAME, name)


def run_training_pipeline() -> dict:
    import hopsworks

    df = load_feature_group()
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    logger.info("Using %d feature columns", len(feature_cols))

    X_train, y_train, X_test, y_test = time_split(df, feature_cols)
    candidates = train_candidates(
        X_train, y_train, X_test, y_test,
        n_features=len(feature_cols), n_targets=len(TARGET_COLUMNS),
    )

    winner_name = min(candidates, key=lambda n: candidates[n][1]["avg_rmse"])
    winner_model, winner_metrics, winner_kind = candidates[winner_name]
    logger.info("Winner: %s (avg RMSE %.3f)", winner_name, winner_metrics["avg_rmse"])

    model_path = save_locally(winner_name, winner_model, winner_kind)

    project = hopsworks.login()  # reuses the same env-based auth as get_feature_store()
    register_in_model_registry(project, winner_name, model_path, winner_metrics)

    return {"winner": winner_name, "metrics": winner_metrics}


def main() -> None:
    try:
        result = run_training_pipeline()
        logger.info("Training pipeline OK. %s", result)
    except Exception:
        logger.exception("Training pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()