"""
Sequence builder for LSTM training.

Converts tabular feature dataframes into 3D tensors of shape
(samples, timesteps, features) suitable for Keras LSTM layers.

Hard rules:
- Sequences are built chronologically. No shuffling inside this module.
- A sample at row i uses features from rows [i-LOOKBACK+1, i] inclusive.
- Rows where the lookback window would go before the dataframe start are dropped.
- Rows where any target is NaN are dropped (must align with has_target flag).
- A window is dropped if it isn't perfectly hourly-contiguous (i.e. it
  silently spans a gap caused by a failed hourly pipeline run). This is
  checked on TIMESTAMPS, not row position, since missing hours show up
  as missing rows, not NaN rows.
- Targets are returned as a 2D array (samples, n_horizons) for multi-output
  training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

LOOKBACK_HOURS = 48


def reindex_to_hourly_grid(
    df: pd.DataFrame,
    interpolate_limit: int = 2,
) -> pd.DataFrame:
    """
    Force the dataframe onto a complete hourly index between its min and
    max timestamp. Any hour missing from Hopsworks (e.g. the hourly
    pipeline failed that run) becomes an explicit NaN row instead of
    silently vanishing.

    Short gaps (<= interpolate_limit hours) are linearly interpolated.
    Longer gaps are left as NaN — build_sequences will refuse to build
    any window that touches them, rather than fabricating data.
    """
    if "timestamp" not in df.columns:
        raise ValueError("df must have a 'timestamp' column")

    df = df.sort_values("timestamp").reset_index(drop=True)
    full_range = pd.date_range(
        df["timestamp"].min(), df["timestamp"].max(), freq="h", tz=df["timestamp"].dt.tz
    )
    df = (
        df.set_index("timestamp")
        .reindex(full_range)
        .rename_axis("timestamp")
        .reset_index()
    )

    n_missing = df.drop(columns="timestamp").isna().any(axis=1).sum()
    logger.info(
        "Reindexed to full hourly grid: %d rows, %d had at least one missing value",
        len(df), n_missing,
    )

    non_ts_cols = [c for c in df.columns if c != "timestamp"]
    df[non_ts_cols] = df[non_ts_cols].interpolate(
        method="linear", limit=interpolate_limit, limit_area="inside"
    )

    still_missing = df.drop(columns="timestamp").isna().any(axis=1).sum()
    if still_missing:
        logger.warning(
            "%d rows still have NaNs after interpolation (gaps longer than "
            "%d hours). These will block any window that touches them.",
            still_missing, interpolate_limit,
        )
    return df


def get_feature_cols(df: pd.DataFrame, target_cols: list[str]) -> list[str]:
    exclude = {"timestamp", "has_target"} | set(target_cols)
    return [c for c in df.columns if c not in exclude]


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    lookback: int = LOOKBACK_HOURS,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Build (X, y, timestamps) tensors for LSTM training.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically by timestamp ascending. Should
        already be passed through reindex_to_hourly_grid() so that gaps
        are explicit NaN rows rather than missing rows.
    feature_cols : list[str]
        Columns to use as model inputs.
    target_cols : list[str]
        Columns to predict (e.g. ["target_24h", "target_48h", "target_72h"]).
    lookback : int
        Number of timesteps in each input sequence.

    Returns
    -------
    X : np.ndarray, shape (n_samples, lookback, n_features)
    y : np.ndarray, shape (n_samples, n_targets)
    timestamps : pd.DatetimeIndex
        The timestamp of each sample's *prediction origin* (the last row
        in its lookback window). Used downstream for chronological splits.
    """
    if df.index.name != "timestamp" and "timestamp" not in df.columns:
        raise ValueError("df must have 'timestamp' as index or column")

    work = df.copy()
    if work.index.name == "timestamp":
        work = work.reset_index()

    if not work["timestamp"].is_monotonic_increasing:
        raise ValueError("df must be sorted by timestamp ascending")

    target_mask = work[target_cols].notna().all(axis=1)
    valid_idx = work.index[target_mask].to_list()

    feature_array = work[feature_cols].to_numpy(dtype=np.float32)
    target_array = work[target_cols].to_numpy(dtype=np.float32)
    timestamp_array = work["timestamp"].to_numpy()

    samples_X, samples_y, samples_ts = [], [], []
    dropped_gap, dropped_nan, dropped_short = 0, 0, 0

    for i in valid_idx:
        if i - lookback + 1 < 0:
            dropped_short += 1
            continue

        ts_window = timestamp_array[i - lookback + 1: i + 1]
        # Contiguity check FIRST: catches missing rows (gaps), which a
        # NaN check alone would never see since missing hours aren't
        # NaN rows unless reindex_to_hourly_grid() was run first.
        deltas = np.diff(ts_window).astype("timedelta64[h]")
        if not np.all(deltas == np.timedelta64(1, "h")):
            dropped_gap += 1
            continue

        window = feature_array[i - lookback + 1: i + 1]
        if np.isnan(window).any():
            dropped_nan += 1
            continue

        samples_X.append(window)
        samples_y.append(target_array[i])
        samples_ts.append(timestamp_array[i])

    logger.info(
        "Window filtering: %d kept, %d dropped (gap-spanning), %d dropped "
        "(NaN feature), %d dropped (insufficient history)",
        len(samples_X), dropped_gap, dropped_nan, dropped_short,
    )

    if not samples_X:
        raise ValueError(
            "No valid sequences could be built. Check lookback, NaN handling, "
            "gap frequency, and target column alignment."
        )

    X = np.stack(samples_X, axis=0)
    y = np.stack(samples_y, axis=0)
    timestamps = pd.DatetimeIndex(samples_ts)

    logger.info(
        "Built %d sequences | X shape %s | y shape %s | first ts %s | last ts %s",
        len(X), X.shape, y.shape, timestamps[0], timestamps[-1],
    )
    return X, y, timestamps