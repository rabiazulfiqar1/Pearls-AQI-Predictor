"""
Register champion + CQR models to Hopsworks Model Registry.

Registers:
  karachi_aqi_champion_{24,48,72}h  — production champion (point prediction).

  karachi_aqi_cqr_{24,48,72}h        — conformal interval system (only when
                                        --champion-only is NOT passed)

Usage:
    python -m models.register_to_registry                  # champion + CQR
    python -m models.register_to_registry --champion-only   # champion only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hopsworks_client import get_model_registry
from utils.logger import get_logger

logger = get_logger(__name__)

CHAMPION_DIR = Path("models/champion")
CQR_DIR = Path("models/cqr")
HORIZONS = [24, 48, 72]


def register_champion(mr, horizon: int, champion_meta: dict) -> None:
    """Register a single horizon's production champion (algorithm varies by horizon)."""
    name = f"karachi_aqi_champion_{horizon}h"
    horizon_dir = CHAMPION_DIR / f"h{horizon}"

    h_data = next(h for h in champion_meta["horizons"] if h["horizon"] == horizon)
    metrics = h_data["metrics"]
    features = h_data["features"]
    model_type = h_data["model_type"]

    description = (
        f"Production champion for Karachi AQI prediction at {horizon}h horizon. "
        f"Algorithm: {model_type}. "
        f"Trained on {champion_meta['n_rows_loaded']} hourly rows from Hopsworks "
        f"feature group aqi_features v1. Top {champion_meta['top_n_shap_features']} "
        f"SHAP features for this horizon: {', '.join(features)}. "
        f"Algorithm + feature ranking chosen from SHAP + ablation holdout comparison "
        f"(Day 9b: Ridge won 24h, XGBoost won 48h/72h); retrained daily on new data "
        f"via GitHub Actions."
    )

    # Store model_type alongside the numeric metrics so it's visible in the
    # registry UI/API without having to open the description text.
    metrics_with_type = {**metrics, "model_type_code": {"ridge": 0, "xgboost": 1}.get(model_type, -1)}

    logger.info("Registering %s (%s) …", name, model_type)
    model = mr.python.create_model(
        name=name,
        metrics=metrics_with_type,
        description=description,
        feature_view=None,  # we use raw feature group
        input_example=None,
    )
    model.save(str(horizon_dir), keep_original_files=True)
    logger.info("  \u2713 Registered %s v%d  (%s, R2=%.4f)", name, model.version, model_type, metrics["r2"])


def register_cqr(mr, horizon: int) -> None:
    """Register a single horizon's CQR (interval) system."""
    name = f"karachi_aqi_cqr_{horizon}h"
    horizon_dir = CQR_DIR / f"h{horizon}"

    with open(horizon_dir / "calibration.json") as f:
        cal = json.load(f)

    metrics = {
        "coverage": cal["holdout_coverage"],
        "avg_width": cal["holdout_avg_width"],
        "Q_widen": cal["Q_widen"],
        "point_r2": cal.get("production_point_r2", cal.get("champion_cqr_holdout_r2", 0)),
    }

    description = (
        f"Conformalized Quantile Regression (Romano et al. 2019, hybrid variant) "
        f"for Karachi AQI prediction intervals at {horizon}h horizon. "
        f"Nominal 80% coverage, achieved {cal['holdout_coverage']*100:.1f}% on holdout. "
        f"Point predictions from production champion (karachi_aqi_champion_{horizon}h); "
        f"intervals from QR + calibration. "
        f"Train/calibration split: {cal['n_train_proper']}/{cal['n_calibration']}."
    )

    logger.info("Registering %s …", name)
    model = mr.python.create_model(
        name=name,
        metrics=metrics,
        description=description,
        feature_view=None,
        input_example=None,
    )
    model.save(str(horizon_dir), keep_original_files=True)
    logger.info(
        "  \u2713 Registered %s v%d  (coverage=%.1f%%)",
        name, model.version, metrics["coverage"] * 100
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champion-only",
        action="store_true",
        help="Register only the champion models, skip CQR. Use this for the daily "
             "pipeline, which only retrains the champion (see models/train_champion.py).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Register models to Hopsworks Model Registry%s", " (champion-only)" if args.champion_only else "")
    logger.info("=" * 60)

    mr = get_model_registry()
    logger.info("Connected to model registry: %s", mr.project_name)

    with open(CHAMPION_DIR / "champion_metadata.json") as f:
        champion_meta = json.load(f)

    for h in HORIZONS:
        register_champion(mr, h, champion_meta)

    if not args.champion_only:
        for h in HORIZONS:
            register_cqr(mr, h)

    print("=" * 60)
    print(f"REGISTRATION COMPLETE — {3 if args.champion_only else 6} models registered (auto-versioned)")
    print("=" * 60)
    print("Champion (point predictions):")
    for h in HORIZONS:
        print(f"  karachi_aqi_champion_{h}h")
    if not args.champion_only:
        print("CQR (interval predictions):")
        for h in HORIZONS:
            print(f"  karachi_aqi_cqr_{h}h")
    print()
    print("View at: https://eu-west.cloud.hopsworks.ai:443/p/32001")


if __name__ == "__main__":
    main()