from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.predict import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    load_latest_row,
    predict_next_3_days,
)
from utils.aqi_calculator import analyze_aqi_alerts, compute_aqi
from utils.hopsworks_client import get_feature_store, get_model_registry

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="wide")

# Styling 
st.markdown(
    """
    <style>
    .metric-card {
        background: #161b2e;
        border: 1px solid #262d45;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: left;
    }
    .metric-card .label { color: #8b93b0; font-size: 0.8rem; margin-bottom: 4px; }
    .metric-card .value { color: #f0f2fa; font-size: 1.8rem; font-weight: 600; line-height: 1.1; }
    .metric-card .sub { font-size: 0.8rem; margin-top: 4px; }
    .alert-card {
        border-radius: 12px;
        border: 1px solid;
        padding: 16px 18px;
        background: rgba(255,255,255,0.03);
    }
    .alert-card .title { font-size: 1.05rem; font-weight: 700; margin-bottom: 4px; }
    .alert-card .body { color: #d6dbeb; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# AQI category thresholds (US EPA) — used for coloring + band shading 
AQI_CATEGORIES = [
    (0, 50, "Good", "#2ecc71"),
    (50, 100, "Moderate", "#f4d03f"),
    (100, 150, "Unhealthy (Sensitive)", "#e67e22"),
    (150, 200, "Unhealthy", "#e74c3c"),
    (200, 300, "Very Unhealthy", "#9b59b6"),
    (300, 500, "Hazardous", "#7b241c"),
]


def aqi_category(value: float) -> tuple[str, str]:
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= value < hi:
            return label, color
    return "Hazardous", AQI_CATEGORIES[-1][3]


def find_column(columns: list[str], keywords: list[str], exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []
    cols_lower = {c: c.lower() for c in columns}
    for kw in keywords:
        for c, cl in cols_lower.items():
            if kw in cl and not any(ex in cl for ex in exclude):
                return c
    return None


def get_current_conditions(latest_row: pd.DataFrame) -> dict[str, Any]:
    cols = list(latest_row.columns)
    row = latest_row.iloc[0]

    temp_col = find_column(cols, ["temperature_2m", "temperature", "temp"])
    humidity_col = find_column(cols, ["relative_humidity", "humidity"])
    pm25_col = find_column(cols, ["pm2_5", "pm25", "pm2p5"])
    pm10_col = find_column(cols, ["pm10"])
    o3_col = find_column(cols, ["o3"])
    co_col = find_column(cols, ["co"], exclude=["cloud"])
    so2_col = find_column(cols, ["so2"])
    no2_col = find_column(cols, ["no2"])
    aqi_col = find_column(cols, ["us_aqi", "aqi"], exclude=["target", "predicted"])

    current_aqi = float(row[aqi_col]) if aqi_col and pd.notna(row[aqi_col]) else None
    if current_aqi is None:
        current_aqi = compute_aqi(
            row[pm25_col] if pm25_col else None,
            row[pm10_col] if pm10_col else None,
            row[o3_col] if o3_col else None,
            row[co_col] if co_col else None,
            row[so2_col] if so2_col else None,
            row[no2_col] if no2_col else None,
        )

    return {
        "temperature": float(row[temp_col]) if temp_col else None,
        "humidity": float(row[humidity_col]) if humidity_col else None,
        "pm25": float(row[pm25_col]) if pm25_col else None,
        "current_aqi": current_aqi,
        "_detected_columns": {
            "temperature": temp_col, "humidity": humidity_col,
            "pm25": pm25_col, "current_aqi": aqi_col,
        },
    }


@st.cache_data(show_spinner=False)
def load_history(days: int = 7) -> pd.DataFrame:
    fs = get_feature_store()
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    return df[df["timestamp"] >= cutoff].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_forecast() -> tuple[pd.DataFrame, dict[str, Any]]:
    latest_row, _fs = load_latest_row()
    mr = get_model_registry()
    predictions = predict_next_3_days(latest_row, mr)
    conditions = get_current_conditions(latest_row)
    return predictions, conditions


st.title("🌫️ Karachi AQI Predictor")
st.caption("Live 3-day forecast — separate champion model per horizon (24h Ridge, 48h/72h XGBoost)")

with st.spinner("Loading latest conditions and forecast..."):
    predictions, conditions = load_forecast()
    alert_analysis = analyze_aqi_alerts(conditions["current_aqi"], predictions.to_dict("records"))

# Current conditions row
c1, c2, c3, c4 = st.columns(4)

with c1:
    if conditions["current_aqi"] is not None:
        label, color = aqi_category(conditions["current_aqi"])
        st.markdown(
            f"""<div class="metric-card">
                <div class="label">Current AQI</div>
                <div class="value">{conditions['current_aqi']:.0f}</div>
                <div class="sub" style="color:{color}">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="metric-card"><div class="label">Current AQI</div>
            <div class="value">N/A</div></div>""",
            unsafe_allow_html=True,
        )

with c2:
    val = conditions["temperature"]
    st.markdown(
        f"""<div class="metric-card">
            <div class="label">Temperature</div>
            <div class="value">{f"{val:.1f}°C" if val is not None else "N/A"}</div>
        </div>""",
        unsafe_allow_html=True,
    )

with c3:
    val = conditions["humidity"]
    st.markdown(
        f"""<div class="metric-card">
            <div class="label">Humidity</div>
            <div class="value">{f"{val:.0f}%" if val is not None else "N/A"}</div>
        </div>""",
        unsafe_allow_html=True,
    )

with c4:
    val = conditions["pm25"]
    st.markdown(
        f"""<div class="metric-card">
            <div class="label">PM2.5</div>
            <div class="value">{f"{val:.1f} µg/m³" if val is not None else "N/A"}</div>
        </div>""",
        unsafe_allow_html=True,
    )

if any(v is None for k, v in conditions.items() if k != "_detected_columns"):
    with st.expander("⚠️ Some current-condition values couldn't be auto-detected"):
        st.write(
            "Column auto-detection didn't find a match for one or more fields. "
            "Detected columns:"
        )
        st.json(conditions["_detected_columns"])
        st.write("If these are wrong or missing, share the actual feature-group column names.")

st.divider()

# ── AQI alert analysis ──────────────────────────────────────────────────
st.subheader("AQI Alert Analysis")

alert_level = alert_analysis["forecast_peak"]["alert_level"]
alert_colors = {
    "normal": "#2ecc71",
    "watch": "#f4d03f",
    "warning": "#e67e22",
    "critical": "#e74c3c",
    "unknown": "#8b93b0",
}
alert_color = alert_colors.get(alert_level, "#8b93b0")

st.markdown(
    f"""<div class="alert-card" style="border-color:{alert_color};">
        <div class="title" style="color:{alert_color};">{alert_analysis['headline']}</div>
        <div class="body">{alert_analysis['current']['advice']}</div>
    </div>""",
    unsafe_allow_html=True,
)

alert_c1, alert_c2, alert_c3 = st.columns(3)

with alert_c1:
    st.metric(
        "Current AQI status",
        alert_analysis["current"]["label"] or "N/A",
        delta=alert_analysis["current"]["alert_level"],
    )
with alert_c2:
    peak = alert_analysis["forecast_peak"]
    peak_label = peak["label"] or "N/A"
    peak_hint = f"{peak['horizon_hours']}h" if peak["horizon_hours"] is not None else None
    st.metric("Worst forecast", peak_label, delta=peak_hint)
with alert_c3:
    st.metric("Alert state", alert_analysis["headline"], delta=alert_analysis["forecast_peak"]["alert_level"])

with st.expander("Forecast alert details"):
    forecast_alerts = pd.DataFrame(alert_analysis["forecast_rows"])
    if not forecast_alerts.empty:
        st.dataframe(
            forecast_alerts[["horizon_hours", "forecast_for", "predicted_aqi", "aqi_label", "alert_level"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No forecast rows available for alert analysis.")

# Forecast trajectory chart 
st.subheader("Forecast Trajectory")

base_time = predictions["generated_at"].iloc[0]
x_vals = [base_time] + predictions["forecast_for"].tolist()
y_vals = [conditions["current_aqi"] if conditions["current_aqi"] is not None else predictions["predicted_aqi"].iloc[0]]
y_vals += predictions["predicted_aqi"].tolist()

upper = [y_vals[0]]
lower = [y_vals[0]]
for _, r in predictions.iterrows():
    rmse = r["holdout_rmse"] if pd.notna(r["holdout_rmse"]) else 0.0
    margin = 1.28 * rmse  # ~80% interval under a normal approximation
    upper.append(r["predicted_aqi"] + margin)
    lower.append(max(0.0, r["predicted_aqi"] - margin))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x_vals + x_vals[::-1], y=upper + lower[::-1],
    fill="toself", fillcolor="rgba(80,140,255,0.15)",
    line={"color": "rgba(0,0,0,0)"}, hoverinfo="skip",
    name="~80% interval (approx., from holdout RMSE)",
))
fig.add_trace(go.Scatter(
    x=x_vals, y=y_vals, mode="lines+markers",
    line={"color": "#5b8dff", "width": 3}, marker={"size": 8},
    name="Predicted AQI",
))
fig.update_layout(
    template="plotly_dark",
    height=420,
    margin={"l": 10, "r": 10, "t": 10, "b": 10},
    xaxis_title="Time",
    yaxis_title="AQI",
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Band is an approximate 80% interval (predicted ± 1.28 × holdout RMSE per horizon), "
    "not a calibrated conformal interval. Register the karachi_aqi_cqr_*h models for "
    "properly calibrated coverage."
)

# Last 7 days chart
st.subheader("Last 7 Days")

history = load_history(days=7)
hist_cols = list(history.columns)
hist_aqi_col = find_column(hist_cols, ["us_aqi", "aqi"], exclude=["target", "predicted"])

if hist_aqi_col is None:
    st.info("Couldn't auto-detect a historical AQI column in the feature group to plot here.")
else:
    fig2 = go.Figure()
    y_max = max(300, float(history[hist_aqi_col].max()) + 20)
    for lo, hi, label, color in AQI_CATEGORIES:
        fig2.add_hrect(y0=lo, y1=min(hi, y_max), fillcolor=color, opacity=0.10, line_width=0)
    fig2.add_trace(go.Scatter(
        x=history["timestamp"], y=history[hist_aqi_col],
        mode="lines", line={"color": "#f0f2fa", "width": 2}, name="Observed AQI",
    ))
    fig2.update_layout(
        template="plotly_dark",
        height=380,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis_title="Time",
        yaxis_title="AQI",
        yaxis_range=[0, y_max],
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

# Forecast table + per-horizon model info 
with st.expander("Forecast details"):
    st.dataframe(
        predictions[["forecast_for", "horizon_hours", "predicted_aqi", "model_type", "holdout_rmse"]],
        width="stretch",
        hide_index=True,
    )