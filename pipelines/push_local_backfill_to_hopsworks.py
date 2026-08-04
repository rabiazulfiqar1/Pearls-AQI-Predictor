"""
Push a locally-saved, already-engineered CSV to Hopsworks.

Exists specifically to resume from Step 3 without re-fetching and
re-engineering ~2 years of data — e.g. after the Step 3 pyarrow/connection
error, where Steps 1-2 already succeeded and
data/processed/karachi_backfill_engineered.csv is sitting there ready to
go. Use this instead of re-running the full backfill_pipeline.py.

Usage:
    python -m pipelines.push_local_backfill_to_hopsworks
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

LOCAL_CSV_PATH = Path("data/processed/karachi_backfill_engineered.csv")


def run_push_pipeline() -> int:
    """
    Load the local engineered CSV and bulk-insert it into Hopsworks.

    Returns the number of rows inserted.
    """
    if not LOCAL_CSV_PATH.exists():
        raise FileNotFoundError(
            f"{LOCAL_CSV_PATH.resolve()} not found. This script expects "
            "backfill_pipeline.py to have already run Steps 1-2 (fetch + "
            "engineer) and written this file — run that first if it "
            "doesn't exist yet."
        )

    logger.info("Step 1/3: loading local engineered CSV")
    engineered = pd.read_csv(LOCAL_CSV_PATH)
    engineered["timestamp"] = (
        pd.to_datetime(engineered["timestamp"])
        .dt.tz_localize(None)
    )

    engineered = engineered.dropna()
    # Convert float columns
    float_cols = [
        "pm2_5",
        "pm10",
        "no2",
        "o3",
        "so2",
        "co",
        "us_aqi",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "pressure_msl",
        "precipitation",
        "cloud_cover",
        "aqi",
        "aqi_lag_24h",
        "pm2_5_lag_24h",
        "aqi_lag_48h",
        "pm2_5_lag_48h",
        "aqi_lag_72h",
        "pm2_5_lag_72h",
        "aqi_change_rate_1h",
        "aqi_change_rate_24h",
        "target_aqi_24h",
        "target_aqi_48h",
        "target_aqi_72h"
    ]

    engineered[float_cols] = engineered[float_cols].astype("float32")


    # Convert integer columns
    int_cols = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "has_target"
    ]

    engineered[int_cols] = engineered[int_cols].astype("int32")
    logger.info("Loaded %d rows from %s", len(engineered), LOCAL_CSV_PATH.resolve())

    logger.info("Step 2/3: connecting to Hopsworks")
    fs = get_feature_store()

    logger.info("Step 3/3: creating/fetching feature group and inserting")
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["timestamp"],
        event_time="timestamp",
        description=(
            "Karachi AQI — hourly engineered features and 24h/48h/72h "
            "forecast targets. Backfilled from Open-Meteo historical "
            "data; kept current by pipelines.feature_pipeline (hourly)."
        ),
        online_enabled=False,
        time_travel_format="HUDI",
    )
    fg.insert(engineered, write_options={"wait_for_job": True})
    logger.info("Push complete: %d rows materialized", len(engineered))
    return len(engineered)


def main() -> None:
    try:
        n_inserted = run_push_pipeline()
        logger.info("Push pipeline OK. Inserted %d rows.", n_inserted)
    except Exception:
        logger.exception("Push pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()