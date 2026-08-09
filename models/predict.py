"""
Prediction pipeline — real-time inference for the next 3 days.

Flow:
    1. Pull the MOST RECENT row from the Hopsworks feature group.
       Feature engineering already computes lags/rolling stats/time
       features at ingestion time (see feature_pipeline.py), so this
       single row already encodes the recent history a model needs.
    2. For EACH horizon (24h/48h/72h), load that horizon's own
       production champion from the Model Registry:
           karachi_aqi_champion_24h  (Ridge)
           karachi_aqi_champion_48h  (XGBoost)
           karachi_aqi_champion_72h  (XGBoost)
       These are separate models with separate SHAP-selected feature
       sets — NOT one multi-output model — matching how
       models/train_champion.py + models/register_to_registry.py
       actually produce and register them.
    3. Predict aqi_t+{h}h per horizon using only that horizon's
       feature subset, filling any missing values with the medians
       saved alongside the model at training time (feature_medians.json).
    4. Map horizons to calendar times and write/print the 3-day forecast.

Usage:
    python -m models.predict
"""

import sys
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store, get_model_registry
from utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

HORIZONS = [24, 48, 72]
CHAMPION_MODEL_NAME_TEMPLATE = "karachi_aqi_champion_{h}h"

PREDICTIONS_FG_NAME = "aqi_predictions"
PREDICTIONS_FG_VERSION = 1


def load_latest_row() -> pd.DataFrame:
    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    latest = df.tail(1).reset_index(drop=True)
    logger.info("Latest feature row timestamp: %s", latest["timestamp"].iloc[0])
    return latest, fs


def load_champion_for_horizon(mr, horizon: int):
    """
    Fetch the currently-registered champion for a single horizon.
    Champions are registered per-horizon (see register_to_registry.py),
    each carrying its own algorithm and its own metrics — pick the
    version with the best (highest) r2, which is the metric
    train_champion.py logs at registration time.
    """
    name = CHAMPION_MODEL_NAME_TEMPLATE.format(h=horizon)
    models = mr.get_models(name=name)
    if not models:
        raise RuntimeError(
            f"No versions of '{name}' found in the Model Registry. "
            f"Run models.train_champion + models.register_to_registry first."
        )
    best = max(models, key=lambda m: m.training_metrics.get("r2", float("-inf")))
    logger.info(
        "Loaded %s v%d (r2=%.3f)", best.name, best.version, best.training_metrics.get("r2", float("nan"))
    )

    model_dir = Path(best.download())
    joblib_path = next(model_dir.glob("*.joblib"), None)
    medians_path = model_dir / "feature_medians.json"

    if joblib_path is None:
        raise RuntimeError(f"No model.joblib found in {model_dir} for {name}")
    if not medians_path.exists():
        raise RuntimeError(f"No feature_medians.json found in {model_dir} for {name}")

    model = joblib.load(joblib_path)

    # feature_medians.json's keys ARE this horizon's top-15 SHAP feature
    # list (it was written from feature_medians.to_dict() in
    # train_champion.py, in the same column order the model was fit on).
    # Reusing it here avoids needing a separate features-list artifact.
    feature_medians = pd.read_json(medians_path, typ="series")

    # model_type_code was logged alongside the numeric metrics in
    # register_to_registry.py ({"ridge": 0, "xgboost": 1}); decode it
    # rather than sniffing type(model).__name__, so this stays correct
    # even if the underlying estimator class changes later.
    code_to_type = {0: "ridge", 1: "xgboost"}
    model_type = code_to_type.get(best.training_metrics.get("model_type_code"), "unknown")
    rmse = best.training_metrics.get("rmse")

    return model, feature_medians, model_type, rmse


def predict_horizon(latest_row: pd.DataFrame, model, feature_medians: pd.Series) -> float:
    features = feature_medians.index.tolist()

    missing_cols = [c for c in features if c not in latest_row.columns]
    if missing_cols:
        raise ValueError(
            f"Latest feature row is missing columns required by this champion: {missing_cols}"
        )

    X = latest_row[features].copy()
    X = X.fillna(feature_medians)

    pred = model.predict(X)
    return float(pred[0])


def predict_next_3_days(latest_row: pd.DataFrame, mr) -> pd.DataFrame:
    base_time = latest_row["timestamp"].iloc[0]
    rows = []
    for h in HORIZONS:
        model, feature_medians, model_type, rmse = load_champion_for_horizon(mr, h)
        predicted_aqi = predict_horizon(latest_row, model, feature_medians)
        rows.append({
            "forecast_for": base_time + timedelta(hours=h),
            "horizon_hours": h,
            "predicted_aqi": predicted_aqi,
            "generated_at": base_time,
            "model_type": model_type,
            "holdout_rmse": rmse,
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

    logger.info("Step 2/3: connecting to model registry")
    mr = get_model_registry()

    logger.info("Step 3/3: predicting next 3 days (per-horizon champions)")
    predictions = predict_next_3_days(latest_row, mr)
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