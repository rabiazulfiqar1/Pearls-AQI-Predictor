"""
Render SHAP/shap_importance_{h}h.csv into report-ready horizontal bar charts,
one PNG per horizon, matching the style: dark title, blue bars, largest at top.

Reads the same files and TOP_N convention as models/train_champion.py
(SHAP_CSV_TEMPLATE, TOP_N), and labels each chart with the selected
algorithm that's
actually champion at that horizon (MODEL_PER_HORIZON), so the titles read
"SHAP Importance — Ridge 24h" / "SHAP Importance — XGBoost 48h" / etc.
without you having to hand-edit them per horizon.

Usage:
    python -m SHAP.plot_shap_importance
    python -m SHAP.plot_shap_importance --top-n 10 --horizons 24 48
    python -m SHAP.plot_shap_importance --shap-dir SHAP --out-dir reports/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Keep these in sync with models/train_champion.py if they ever change there.
TOP_N = 15
MODEL_PER_HORIZON = {24: "Ridge", 48: "XGBoost", 72: "XGBoost"}
DEFAULT_HORIZONS = [24, 48, 72]

BAR_COLOR = "#0d8bf2"  # matches the reference screenshot's blue


def load_top_features(csv_path: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "feature" not in df.columns:
        raise ValueError(f"{csv_path} has no 'feature' column. Columns found: {list(df.columns)}")

    value_col = None
    for candidate in ("mean_abs_shap", "importance", "shap_importance"):
        if candidate in df.columns:
            value_col = candidate
            break
    if value_col is None:
        # Fall back to the first non-'feature' numeric column.
        numeric_cols = [c for c in df.columns if c != "feature" and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError(f"{csv_path} has no recognizable SHAP-value column. Columns found: {list(df.columns)}")
        value_col = numeric_cols[0]

    df = df[["feature", value_col]].rename(columns={value_col: "mean_abs_shap"})
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_n)
    return df


def plot_horizon(df: pd.DataFrame, horizon: int, model_label: str, out_path: Path) -> None:
    # barh plots bottom-to-top, so reverse so the largest bar ends up on top.
    df_plot = df.iloc[::-1]

    fig_height = max(4, 0.4 * len(df_plot) + 1.2)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    ax.barh(df_plot["feature"], df_plot["mean_abs_shap"], color=BAR_COLOR)
    ax.set_title(f"SHAP Importance — {model_label} {horizon}h", fontsize=14, pad=14)
    ax.set_xlabel("mean |SHAP value|")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shap-dir", type=Path, default=Path("SHAP"),
                         help="Directory containing shap_importance_{h}h.csv files (default: SHAP)")
    parser.add_argument("--out-dir", type=Path, default=Path("SHAP"),
                         help="Directory to write PNGs to (default: SHAP)")
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS,
                         help="Horizons to render (default: 24 48 72)")
    parser.add_argument("--top-n", type=int, default=TOP_N,
                         help=f"Number of top SHAP features to plot per horizon (default: {TOP_N})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for h in args.horizons:
        csv_path = args.shap_dir / f"shap_importance_{h}h.csv"
        if not csv_path.exists():
            print(f"  ! skipping {h}h — {csv_path} not found")
            continue

        model_label = MODEL_PER_HORIZON.get(h, "")
        df = load_top_features(csv_path, args.top_n)
        out_path = args.out_dir / f"shap_importance_{h}h.png"
        plot_horizon(df, h, model_label, out_path)
        print(f"  \u2713 wrote {out_path}  (top {len(df)} of {csv_path.name})")


if __name__ == "__main__":
    main()