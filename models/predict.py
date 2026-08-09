"""
Prediction pipeline — real-time inference for the next 3 days.

Flow:
    1. Pull the MOST RECENT row from the Hopsworks feature group.
       Because feature engineering already computes lags/rolling
       stats/time features at ingestion time (see feature_pipeline.py),
       this single row already encodes the recent history a model
       needs — no need to reconstruct a window here.
    2. Load the current-best model from the Hopsworks Model Registry
       (the one train.py registered).
    3. Predict [aqi_t+24h, aqi_t+48h, aqi_t+72h] from that row.
    4. Map horizons to calendar days and write/print the 3-day
       forecast. 

Usage:
    python -m models.predict
"""

import sys
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
MODEL_NAME = "aqi_multi_horizon_model"

PREDICTIONS_FG_NAME = "aqi_predictions"
PREDICTIONS_FG_VERSION = 1

# --- must match training_pipeline.py exactly ---------------------------
TARGET_COLUMNS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
NON_FEATURE_COLUMNS = ["timestamp"] + TARGET_COLUMNS
HORIZON_HOURS = {"target_aqi_24h": 24, "target_aqi_48h": 48, "target_aqi_72h": 72}
# -------------------------------------------------------------------------


def load_latest_row() -> pd.DataFrame:
    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    latest = df.tail(1).reset_index(drop=True)
    logger.info("Latest feature row timestamp: %s", latest["timestamp"].iloc[0])
    return latest, fs


def load_best_model():
    """
    Fetch the currently-registered model from the Hopsworks Model
    Registry. Picks the version with the lowest avg_rmse metric (the
    metric train.py logs at registration time).
    """
    import hopsworks

    project = hopsworks.login()  # same auth path as train.py
    mr = project.get_model_registry()
    models = mr.get_models(name=MODEL_NAME)
    if not models:
        raise RuntimeError(
            f"No versions of '{MODEL_NAME}' found in the Model Registry. "
            f"Run train.py at least once first."
        )
    best = min(models, key=lambda m: m.training_metrics.get("avg_rmse", float("inf")))
    logger.info("Loaded model '%s' v%d (avg_rmse=%.3f)",
                best.name, best.version, best.training_metrics.get("avg_rmse", -1))

    model_dir = Path(best.download())
    joblib_path = next(model_dir.glob("*.joblib"), None)
    keras_path = next(model_dir.glob("*.keras"), None)

    if joblib_path:
        return joblib.load(joblib_path), "sklearn"
    elif keras_path:
        from keras.models import load_model
        return load_model(keras_path), "keras"
    else:
        raise RuntimeError(f"No model file found in {model_dir}")


def predict_next_3_days(latest_row: pd.DataFrame, model, kind: str) -> pd.DataFrame:
    feature_cols = [c for c in latest_row.columns if c not in NON_FEATURE_COLUMNS]
    X = latest_row[feature_cols].values

    preds = model.predict(X)
    preds = preds[0] if preds.ndim > 1 else preds  # single row in, single row out

    base_time = latest_row["timestamp"].iloc[0]
    rows = []
    for target_col, hours_ahead in HORIZON_HOURS.items():
        idx = TARGET_COLUMNS.index(target_col)
        rows.append({
            "forecast_for": base_time + timedelta(hours=hours_ahead),
            "horizon_hours": hours_ahead,
            "predicted_aqi": float(preds[idx]),
            "generated_at": base_time,
        })
    result = pd.DataFrame(rows)
    logger.info("3-day forecast:\n%s", result.to_string(index=False))
    return result


def log_predictions(fs, predictions: pd.DataFrame) -> None:
    """Optional: store predictions in their own feature group so you
    can later join against realized AQI and monitor drift."""
    fg = fs.get_or_create_feature_group(
        name=PREDICTIONS_FG_NAME,
        version=PREDICTIONS_FG_VERSION,
        primary_key=["generated_at", "horizon_hours"],
        event_time="generated_at",
        description="Logged 24h/48h/72h AQI predictions, for later comparison against realized AQI.",
        online_enabled=False,
    )
    fg.insert(predictions, write_options={"wait_for_job": True})
    logger.info("Logged %d prediction rows to '%s'", len(predictions), PREDICTIONS_FG_NAME)


def run_prediction_pipeline() -> pd.DataFrame:
    logger.info("Step 1/3: fetching latest feature row")
    latest_row, fs = load_latest_row()

    logger.info("Step 2/3: loading best registered model")
    model, kind = load_best_model()

    logger.info("Step 3/3: predicting next 3 days")
    predictions = predict_next_3_days(latest_row, model, kind)
    log_predictions(fs, predictions)

    return predictions


def main() -> None:
    try:
        predictions = run_prediction_pipeline()
        logger.info("Prediction pipeline OK.\n%s", predictions.to_string(index=False))
    except Exception:
        logger.exception("Prediction pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()