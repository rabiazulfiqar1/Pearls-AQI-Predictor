"""
Backfill pipeline — one-time historical seed for Hopsworks.

Run this ONCE (or whenever you need to fully rebuild the feature group)
before the hourly pipeline (pipelines/feature_pipeline.py) can do its
job — the hourly job only inserts *new* rows relative to what's already
in Hopsworks, so Hopsworks needs a real history to compare against
first.

Flow:
    1. Fetch ~1 year of historical air quality + weather from
       Open-Meteo, chunked by year (same approach as the exploration
       script, but chunked so a single oversized request doesn't risk
       timing out or getting silently truncated).
    2. Clean + engineer features (utils.features — identical code path
       the hourly pipeline uses, so backfilled rows and hourly rows are
       byte-for-byte the same shape).
    3. Create the Hopsworks feature group (if it doesn't exist yet) and
       bulk-insert the engineered history.

Also writes a local CSV backup to data/processed/ — per the project's
existing convention of keeping local backups for the backfill case,
even though Hopsworks is the source of truth going forward.

Usage:
    python -m pipelines.backfill_pipeline
"""

import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import (
    API_RETRY_BASE_DELAY,
    API_RETRY_COUNT,
    KARACHI_LAT,
    KARACHI_LON,
)
from utils.features import clean_raw_data, engineer_features
from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

AIR_QUALITY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
WEATHER_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

END_DATE = datetime.now(timezone.utc).date() - timedelta(days=1)
START_DATE = date(2025, 7, 1)

REQUEST_DELAY_SECONDS = 1.0

LOCAL_BACKUP_PATH = Path("data/processed/karachi_backfill_engineered.csv")


def _make_client() -> openmeteo_requests.Client:
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(
        cache_session,
        retries=API_RETRY_COUNT,
        backoff_factor=API_RETRY_BASE_DELAY / 10,
    )
    return openmeteo_requests.Client(session=retry_session)


def year_chunks(start: datetime.date, end: datetime.date):
    chunk_start = start
    while chunk_start <= end:
        try:
            chunk_end = chunk_start.replace(year=chunk_start.year + 1) - pd.Timedelta(days=1)
        except ValueError:
            chunk_end = chunk_start.replace(year=chunk_start.year + 1, day=28) - pd.Timedelta(days=1)
        chunk_end = min(chunk_end.date() if hasattr(chunk_end, "date") else chunk_end, end)
        yield chunk_start, chunk_end
        chunk_start = chunk_end + pd.Timedelta(days=1)
        chunk_start = chunk_start.date() if hasattr(chunk_start, "date") else chunk_start


def fetch_air_quality_chunk(client, start, end) -> pd.DataFrame:
    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "start_date": str(start),
        "end_date": str(end),
        "timezone": "UTC",
        "hourly": ["pm2_5", "pm10", "nitrogen_dioxide", "ozone",
                   "sulphur_dioxide", "carbon_monoxide", "us_aqi"],
    }
    hourly = client.weather_api(AQ_URL, params=params)[0].Hourly()
    return pd.DataFrame({
        "timestamp": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left",
        ),
        "pm2_5":  hourly.Variables(0).ValuesAsNumpy(),
        "pm10":   hourly.Variables(1).ValuesAsNumpy(),
        "no2":    hourly.Variables(2).ValuesAsNumpy(),
        "o3":     hourly.Variables(3).ValuesAsNumpy(),
        "so2":    hourly.Variables(4).ValuesAsNumpy(),
        "co":     hourly.Variables(5).ValuesAsNumpy(),
        "us_aqi": hourly.Variables(6).ValuesAsNumpy(),
    })


def fetch_weather_chunk(client, start, end) -> pd.DataFrame:
    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "start_date": str(start),
        "end_date": str(end),
        "timezone": "UTC",
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                   "pressure_msl", "precipitation", "cloud_cover"],
    }
    hourly = client.weather_api(WEATHER_URL, params=params)[0].Hourly()
    return pd.DataFrame({
        "timestamp": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()), inclusive="left",
        ),
        "temperature_2m":       hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed_10m":       hourly.Variables(2).ValuesAsNumpy(),
        "pressure_msl":         hourly.Variables(3).ValuesAsNumpy(),
        "precipitation":        hourly.Variables(4).ValuesAsNumpy(),
        "cloud_cover":          hourly.Variables(5).ValuesAsNumpy(),
    })


def fetch_historical_data() -> pd.DataFrame:
    client = _make_client()
    chunks = list(year_chunks(START_DATE, END_DATE))
    logger.info("Fetching %d chunk(s): %s to %s", len(chunks), START_DATE, END_DATE)

    aq_frames, weather_frames = [], []
    for i, (c_start, c_end) in enumerate(chunks, start=1):
        logger.info("[%d/%d] %s to %s", i, len(chunks), c_start, c_end)
        aq_frames.append(fetch_air_quality_chunk(client, c_start, c_end))
        time.sleep(REQUEST_DELAY_SECONDS)
        weather_frames.append(fetch_weather_chunk(client, c_start, c_end))
        time.sleep(REQUEST_DELAY_SECONDS)

    aq_df = pd.concat(aq_frames, ignore_index=True).drop_duplicates(subset="timestamp")
    weather_df = pd.concat(weather_frames, ignore_index=True).drop_duplicates(subset="timestamp")
    merged = pd.merge(aq_df, weather_df, on="timestamp", how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)


def run_backfill_pipeline() -> int:
    """
    Fetch, engineer, and bulk-insert the full historical window into
    Hopsworks. Returns the number of rows inserted.
    """
    logger.info("Step 1/4: fetching historical data from %s to %s",START_DATE,END_DATE)
    raw_df = fetch_historical_data()
    logger.info("Raw rows fetched: %d", len(raw_df))

    logger.info("Step 2/4: cleaning + engineering features")
    cleaned = clean_raw_data(raw_df)
    engineered = engineer_features(cleaned)
    logger.info("Engineered rows: %d (of %d raw)", len(engineered), len(raw_df))

    LOCAL_BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(LOCAL_BACKUP_PATH, index=False)
    logger.info("Local backup written: %s", LOCAL_BACKUP_PATH.resolve())

    logger.info("Step 3/4: connecting to Hopsworks")
    fs = get_feature_store()

    logger.info("Step 4/4: creating/fetching feature group and inserting")
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
    )
    fg.insert(engineered, write_options={"wait_for_job": True})
    logger.info("Backfill insert complete: %d rows materialized", len(engineered))
    return len(engineered)


def main() -> None:
    try:
        n_inserted = run_backfill_pipeline()
        logger.info("Backfill pipeline OK. Inserted %d rows.", n_inserted)
    except Exception as exc:
        logger.exception("Backfill pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
