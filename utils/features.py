"""
Data cleaning and feature engineering for the Pearls AQI Predictor pipeline.

Two public entry points:

    clean_raw_data(df)       — drops sensor errors, clips physical limits,
                               forward-fills short gaps, dedupes.

    engineer_features(df)    — adds time, rolling, lag, derived, and
                               target columns. Expects already-cleaned data
                               with a 'timestamp' column (UTC, hourly).

Both are idempotent: running twice on the same DataFrame is safe.

All configurable constants come from config.config — never hardcoded here.
"""

import numpy as np
import pandas as pd

from config.config import (
    CHANGE_RATE_WINDOWS_HOURS,
    LAG_HOURS,
    ROLLING_WINDOW_BASELINE,
    ROLLING_WINDOW_LONG,
    ROLLING_WINDOW_SHORT,
    TARGET_HORIZONS_HOURS,
)

POLLUTANTS: list[str] = ["pm2_5", "pm10", "no2", "o3", "so2", "co"]

# Matches what fetch_karachi_data.py / build_aqi_features.py actually pull
# from Open-Meteo's weather API — not the generic "temperature/humidity/
# wind_speed/pressure" names, so no silent rename step is needed upstream.
WEATHER: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "pressure_msl",
    "precipitation",
    "cloud_cover",
]

# Physical upper bounds for pollutant clipping. Anything above these is
# almost certainly a sensor error, not real air.
POLLUTANT_CAPS = {
    "pm2_5": 1000,
    "pm10":  1500,
    # Other pollutants left uncapped — their ranges vary enough that a
    # blanket cap would cut into real values.
}


# ========================================================================
# Cleaning
# ========================================================================

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw pollutant/weather data from Open-Meteo.

    Steps:
        1. Drop rows with missing or invalid timestamps.
        2. Sort by timestamp ascending.
        3. Clip negative pollutant readings to 0 (sensor errors).
        4. Cap unreasonably high pollutant readings at physical limits.
        5. Forward-fill short gaps (<= 3 hours) in pollutant columns.
        6. Drop rows where any core pollutant is still missing.
        7. Deduplicate by timestamp (keep last — latest fetch wins).

    The input DataFrame is not mutated.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()

    if "timestamp" not in out.columns:
        raise ValueError("clean_raw_data: 'timestamp' column is required")

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    present_pollutants = [c for c in POLLUTANTS if c in out.columns]

    for col in present_pollutants:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].clip(lower=0)
        if col in POLLUTANT_CAPS:
            out[col] = out[col].clip(upper=POLLUTANT_CAPS[col])

    if present_pollutants:
        out[present_pollutants] = out[present_pollutants].ffill(limit=3)
        out = out.dropna(subset=present_pollutants)

    out = out.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    return out


# ========================================================================
# Feature engineering
# ========================================================================

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype("int16")
    df["day_of_week"] = ts.dt.dayofweek.astype("int16")
    df["month"] = ts.dt.month.astype("int16")
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")
    # Cyclical encodings so hour 23 and hour 0 (and Dec/Jan) read as
    # adjacent, not maximally far apart.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _add_aqi_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use Open-Meteo's us_aqi field directly as the AQI value, rather than
    recomputing it from pollutant concentrations.

    Open-Meteo's us_aqi is already derived from the same underlying CAMS
    pollutant data via the EPA formula, so a local recomputation would
    mostly just re-derive the same number while adding a slow row-wise
    apply() and a second place the unit-conversion/breakpoint logic could
    silently disagree with the source. utils/aqi_calculator.compute_aqi()
    is kept around as a fallback for any future data source that only
    reports raw pollutant concentrations, and get_category()/
    get_category_label() (re-exported above) are still used for alerting
    and dashboard labeling regardless of where the numeric AQI came from.
    """
    if "us_aqi" not in df.columns:
        raise ValueError(
            "_add_aqi_column: 'us_aqi' column is required (expected from "
            "Open-Meteo's air-quality API)"
        )
    df["aqi"] = df["us_aqi"]
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df["aqi_rolling_6h"] = (
        df["aqi"].rolling(window=ROLLING_WINDOW_SHORT, min_periods=1).mean()
    )
    df["aqi_rolling_24h"] = (
        df["aqi"].rolling(window=ROLLING_WINDOW_LONG, min_periods=1).mean()
    )
    df["rolling_30day_avg"] = (
        df["aqi"].rolling(window=ROLLING_WINDOW_BASELINE, min_periods=1).mean()
    )
    df["rolling_30day_std"] = (
        df["aqi"].rolling(window=ROLLING_WINDOW_BASELINE, min_periods=2).std()
    )
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in LAG_HOURS:
        df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)
        df[f"pm2_5_lag_{lag}h"] = df["pm2_5"].shift(lag)
    return df


def _add_change_rate(df: pd.DataFrame) -> pd.DataFrame:
    # Config-driven, multi-window — the original single diff() is the
    # special case of this with window=1.
    for w in CHANGE_RATE_WINDOWS_HOURS:
        df[f"aqi_change_rate_{w}h"] = (df["aqi"] - df["aqi"].shift(w)) / w
    return df


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add target columns by shifting AQI *backwards* in time. The row at
    timestamp t gets target_aqi_24h = AQI at t+24h, etc.

    Rows near the end of the series will have NaN targets; these are
    flagged by `has_target` and filtered out during training but retained
    for serving (they're the rows we actually want to predict from).
    """
    for h in TARGET_HORIZONS_HOURS:
        df[f"target_aqi_{h}h"] = df["aqi"].shift(-h)

    target_cols = [f"target_aqi_{h}h" for h in TARGET_HORIZONS_HOURS]
    df["has_target"] = df[target_cols].notna().all(axis=1).astype("int8")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Input:  cleaned DataFrame with columns:
        timestamp (UTC), us_aqi, pm2_5, pm10, no2, o3, so2, co,
        temperature_2m, relative_humidity_2m, wind_speed_10m,
        pressure_msl, precipitation, cloud_cover

    Output: same DataFrame plus all engineered columns.

    The input DataFrame is not mutated.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    required = {"timestamp", "us_aqi", *POLLUTANTS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"engineer_features: missing required columns: {sorted(missing)}"
        )

    out = df.copy()
    out = out.sort_values("timestamp").reset_index(drop=True)

    out = _add_time_features(out)
    out = _add_aqi_column(out)
    out = _add_rolling_features(out)
    out = _add_lag_features(out)
    out = _add_change_rate(out)
    out = _add_targets(out)

    return out


# ========================================================================
# Convenience — the List of feature columns the models will consume
# ========================================================================

def get_feature_columns() -> list[str]:
    """
    Return the ordered List of input feature column names (not targets,
    not metadata). The training pipeline uses this to slice X.
    """
    feature_cols = (
        POLLUTANTS
        + WEATHER
        + ["hour", "day_of_week", "month", "is_weekend",
           "hour_sin", "hour_cos", "month_sin", "month_cos"]
        + ["aqi_rolling_6h", "aqi_rolling_24h"]
        + [f"aqi_lag_{h}h" for h in LAG_HOURS]
        + [f"pm2_5_lag_{h}h" for h in LAG_HOURS]
        + ["aqi"]
        + [f"aqi_change_rate_{w}h" for w in CHANGE_RATE_WINDOWS_HOURS]
        + ["rolling_30day_avg", "rolling_30day_std"]
    )
    return feature_cols


def get_target_columns() -> list[str]:
    """Return the target column names in horizon order."""
    return [f"target_aqi_{h}h" for h in TARGET_HORIZONS_HOURS]