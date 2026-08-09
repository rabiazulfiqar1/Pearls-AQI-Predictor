"""
Daily champion retraining — mixed-algorithm karachi_aqi production model.

Retrains the production champion per horizon on the latest data in
Hopsworks, using each horizon's own top-15 SHAP-ranked feature set from
shap_pruned_classical_models.ipynb. The algorithm is NOT the same across
horizons — it's picked per horizon to match what actually won on your
holdout results:

    24h -> Ridge     (R2 0.516 vs XGBoost's 0.504 — Ridge wins clearly)
    48h -> XGBoost    (R2 0.150 vs Ridge's 0.087)
    72h -> XGBoost    (R2 -0.019 vs Ridge's -0.024 — closer, but still ahead)

Feature sets are loaded from SHAP/shap_importance_{h}h.csv (checked
into the repo) — NOT recomputed here. XGBoost's hyperparameters are also
FIXED (see XGB_FIXED_PARAMS), not searched here — a RandomizedSearchCV in
the daily job (60 fits/horizon) caused timeout on GitHub Actions. 
Re-deriving features, tuning hyperparameters is a periodic, manual step done in the Kaggle
notebooks — daily retraining is about absorbing new data with settings
that are already settled, nothing more. This keeps the whole job to a
handful of fast model fits.

Writes model artifacts + champion_metadata.json to models/champion/h{h}/
in the layout models/register_to_registry.py expects, so the two chain
directly in the daily GitHub Actions job:

    python -m models.train_champion
    python -m models.register_to_registry --champion-only

Usage:
    python -m models.train_champion
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_feature_store
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Config ───────────────────────────────────────────────────────────────
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

HORIZONS = [24, 48, 72]
TOP_N = 15  # matches shap_pruned_classical_models.ipynb (Day 9b)

# Which algorithm is champion at each horizon — picked from your actual
# holdout results (see module docstring), not a fixed choice across the board.
MODEL_PER_HORIZON = {24: "ridge", 48: "xgboost", 72: "xgboost"}

SHAP_CSV_TEMPLATE = str(Path(__file__).resolve().parent.parent / "SHAP" / "shap_importance_{h}h.csv")

TRAIN_FRACTION = 0.8          # chronological — last 20% is holdout, matches the notebook
CV_N_SPLITS = 5
RIDGE_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0]  # same grid as fit_ridge() in the notebook

XGB_FIXED_PARAMS = {
    48: {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 1.0,
        "min_child_weight": 3,
    },
    72: {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 1.0,
        "min_child_weight": 3,
    },
}

CHAMPION_DIR = Path("models/champion")

# Guard: refuse to overwrite a previously-registered champion with a
# meaningfully worse one (e.g. a bad day of ingested data). Set to None
# to disable. Intentionally generous — a sanity check against broken
# data, not a tight quality gate.
MAX_R2_REGRESSION = 0.15


def load_horizon_features(h: int) -> List[str]:
    path = Path(SHAP_CSV_TEMPLATE.format(h=h))
    if not path.exists():
        raise FileNotFoundError(
            f"Missing SHAP ranking for {h}h horizon: {path}. "
            f"Expected a checked-in shap_importance_{h}h.csv (top-{TOP_N} features used)."
        )
    ranking = pd.read_csv(path)
    features = ranking["feature"].head(TOP_N).tolist()
    logger.info("  %dh feature set (top %d SHAP): %s", h, TOP_N, features)
    return features


def load_training_data(all_features: set) -> pd.DataFrame:
    logger.info("Connecting to Hopsworks and reading '%s' v%d", FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)
    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing = [c for c in all_features if c not in df.columns]
    if missing:
        raise ValueError(
            f"SHAP-selected feature(s) not found in feature group: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    if "has_target" in df.columns:
        df = df[df["has_target"] == 1].copy()

    logger.info("Loaded %d rows with targets", len(df))
    return df


def chronological_split(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_fraction)
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_ridge(X_train: pd.DataFrame, y_train: pd.Series):
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    cv_splits = list(TimeSeriesSplit(n_splits=CV_N_SPLITS).split(np.arange(len(X_train))))
    search = GridSearchCV(
        pipe,
        param_grid={"ridge__alpha": RIDGE_ALPHA_GRID},
        cv=cv_splits,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    logger.info("    best alpha=%s (cv RMSE=%.3f)", search.best_params_["ridge__alpha"], -search.best_score_)
    return search.best_estimator_


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series, horizon: int):
    # No scaler — tree models don't need one. No search — see XGB_FIXED_PARAMS
    # above for why: this is a single fit with pre-tuned hyperparameters,
    # which is what keeps the daily job fast.
    params = XGB_FIXED_PARAMS[horizon]
    model = XGBRegressor(
        objective="reg:squarederror", random_state=42, n_jobs=-1, tree_method="hist", **params,
    )
    model.fit(X_train, y_train)
    return model


# fit_ridge(X_train, y_train) / fit_xgboost(X_train, y_train, horizon) have
# different signatures (XGBoost needs the horizon to look up its fixed
# params), so dispatch by hand in run_training() rather than a flat dict of
# interchangeable callables.
FITTERS = {"ridge": lambda X, y, h: fit_ridge(X, y), "xgboost": fit_xgboost}


def load_previous_metrics() -> Dict[int, dict]:
    """Best-effort read of the last run's metrics, for the regression guard.
    Returns {} if there's no previous run (e.g. first-ever daily run)."""
    meta_path = CHAMPION_DIR / "champion_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return {h["horizon"]: h["metrics"] for h in meta["horizons"]}
    except Exception:
        logger.warning("Could not parse previous champion_metadata.json — skipping regression guard for this run")
        return {}


def run_training() -> None:
    horizon_features = {h: load_horizon_features(h) for h in HORIZONS}
    all_features = set().union(*horizon_features.values())

    df = load_training_data(all_features)
    train_df, holdout_df = chronological_split(df, TRAIN_FRACTION)
    logger.info("Split — train: %d  holdout: %d", len(train_df), len(holdout_df))

    previous_metrics = load_previous_metrics()

    horizons_meta: List[dict] = []
    failures: List[str] = []

    for h in HORIZONS:
        features = horizon_features[h]
        model_type = MODEL_PER_HORIZON[h]
        fit_fn = FITTERS[model_type]
        target_col = f"target_aqi_{h}h"
        if target_col not in df.columns:
            raise ValueError(f"No target column found for horizon {h} (expected {target_col})")

        X_train_full = train_df[features].copy()
        X_holdout_full = holdout_df[features].copy()
        feature_medians = X_train_full.median()
        X_train_full = X_train_full.fillna(feature_medians)
        X_holdout_full = X_holdout_full.fillna(feature_medians)

        y_train = train_df[target_col]
        y_holdout = holdout_df[target_col]
        train_mask = y_train.notna()
        holdout_mask = y_holdout.notna()

        X_train = X_train_full[train_mask]
        y_train = y_train[train_mask]
        X_holdout = X_holdout_full[holdout_mask]
        y_holdout = y_holdout[holdout_mask]

        logger.info(
            "=== Horizon %dh — retraining %s on %d rows, %d SHAP features ===",
            h, model_type, len(X_train), len(features),
        )
        t0 = time.time()
        model = fit_fn(X_train, y_train, h)
        fit_time = time.time() - t0
        preds = model.predict(X_holdout)
        metrics = regression_metrics(y_holdout, preds)
        logger.info(
            "  %dh (%s)  fit_time=%.1fs  RMSE=%.3f  MAE=%.3f  R2=%.3f",
            h, model_type, fit_time, metrics["rmse"], metrics["mae"], metrics["r2"],
        )

        prev = previous_metrics.get(h)
        if prev is not None and MAX_R2_REGRESSION is not None:
            drop = prev["r2"] - metrics["r2"]
            if drop > MAX_R2_REGRESSION:
                failures.append(
                    f"{h}h ({model_type}): R2 dropped {drop:.3f} vs previous run "
                    f"({prev['r2']:.3f} -> {metrics['r2']:.3f}), exceeds MAX_R2_REGRESSION={MAX_R2_REGRESSION}"
                )

        horizon_dir = CHAMPION_DIR / f"h{h}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, horizon_dir / "model.joblib")
        with open(horizon_dir / "feature_medians.json", "w") as f:
            json.dump(feature_medians.to_dict(), f, indent=2)

        horizons_meta.append({
            "horizon": h,
            "model_type": model_type,
            "metrics": metrics,
            "n_train": len(X_train),
            "n_holdout": len(X_holdout),
            "features": features,
        })

    champion_metadata = {
        "n_rows_loaded": len(df),
        "train_fraction": TRAIN_FRACTION,
        "top_n_shap_features": TOP_N,
        "horizons": horizons_meta,
    }
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHAMPION_DIR / "champion_metadata.json", "w") as f:
        json.dump(champion_metadata, f, indent=2)
    logger.info("Wrote champion_metadata.json — ready for models.register_to_registry")

    if failures:
        # Artifacts are still written to disk for inspection, but we exit
        # non-zero so the workflow fails BEFORE the register_to_registry
        # step runs — a regressed model should never reach production.
        for f_msg in failures:
            logger.error("REGRESSION GUARD TRIGGERED: %s", f_msg)
        raise RuntimeError(
            "One or more horizons regressed beyond MAX_R2_REGRESSION vs the previous "
            "champion run. Not proceeding to registry. See errors above."
        )


def main() -> None:
    try:
        run_training()
    except Exception:
        logger.exception("Champion training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()