# from __future__ import annotations

# import sys
# from pathlib import Path
# from typing import Any

# import pandas as pd
# import plotly.graph_objects as go
# import streamlit as st

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# from models.predict import (
#     FEATURE_GROUP_NAME,
#     FEATURE_GROUP_VERSION,
#     load_latest_row,
#     predict_next_3_days,
# )
# from utils.aqi_calculator import analyze_aqi_alerts, compute_aqi
# from utils.hopsworks_client import get_feature_store, get_model_registry

# st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="wide")

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# EVALUATION_RESULTS = PROJECT_ROOT / "results" / "shap_pruned_results.csv"
# SHAP_DIRECTORY = PROJECT_ROOT / "SHAP"

# # Styling 
# st.markdown(
#     """
#     <style>
#     .stApp { background: #0f1117; color: #f0f2fa; }
#     [data-testid="stHeader"] { background: #0f1117; }
#     [data-testid="stSidebar"] { background: #171a21; }
#     h1, h2, h3, p, label { color: #f0f2fa; }
#     .metric-card {
#         background: #171a21;
#         border: 1px solid #343a46;
#         border-radius: 10px;
#         padding: 16px 20px;
#         text-align: left;
#     }
#     .metric-card .label { color: #a3acbd; font-size: 0.8rem; margin-bottom: 4px; }
#     .metric-card .value { color: #f0f2fa; font-size: 1.8rem; font-weight: 600; line-height: 1.1; }
#     .metric-card .sub { font-size: 0.8rem; margin-top: 4px; }
#     .alert-card {
#         border-radius: 12px;
#         border: 1px solid;
#         padding: 16px 18px;
#         background: #171a21;
#     }
#     .alert-card .title { font-size: 1.05rem; font-weight: 700; margin-bottom: 4px; }
#     .alert-card .body { color: #d6dbeb; font-size: 0.95rem; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # AQI category thresholds (US EPA) — used for coloring + band shading 
# AQI_CATEGORIES = [
#     (0, 50, "Good", "#2ecc71"),
#     (50, 100, "Moderate", "#f4d03f"),
#     (100, 150, "Unhealthy (Sensitive)", "#e67e22"),
#     (150, 200, "Unhealthy", "#e74c3c"),
#     (200, 300, "Very Unhealthy", "#9b59b6"),
#     (300, 500, "Hazardous", "#7b241c"),
# ]


# def aqi_category(value: float) -> tuple[str, str]:
#     for lo, hi, label, color in AQI_CATEGORIES:
#         if lo <= value < hi:
#             return label, color
#     return "Hazardous", AQI_CATEGORIES[-1][3]


# def find_column(columns: list[str], keywords: list[str], exclude: list[str] | None = None) -> str | None:
#     exclude = exclude or []
#     cols_lower = {c: c.lower() for c in columns}
#     for kw in keywords:
#         for c, cl in cols_lower.items():
#             if kw in cl and not any(ex in cl for ex in exclude):
#                 return c
#     return None


# def get_current_conditions(latest_row: pd.DataFrame) -> dict[str, Any]:
#     cols = list(latest_row.columns)
#     row = latest_row.iloc[0]

#     temp_col = find_column(cols, ["temperature_2m", "temperature", "temp"])
#     humidity_col = find_column(cols, ["relative_humidity", "humidity"])
#     pm25_col = find_column(cols, ["pm2_5", "pm25", "pm2p5"])
#     pm10_col = find_column(cols, ["pm10"])
#     o3_col = find_column(cols, ["o3"])
#     co_col = find_column(cols, ["co"], exclude=["cloud"])
#     so2_col = find_column(cols, ["so2"])
#     no2_col = find_column(cols, ["no2"])
#     aqi_col = find_column(cols, ["us_aqi", "aqi"], exclude=["target", "predicted"])

#     current_aqi = float(row[aqi_col]) if aqi_col and pd.notna(row[aqi_col]) else None
#     if current_aqi is None:
#         current_aqi = compute_aqi(
#             row[pm25_col] if pm25_col else None,
#             row[pm10_col] if pm10_col else None,
#             row[o3_col] if o3_col else None,
#             row[co_col] if co_col else None,
#             row[so2_col] if so2_col else None,
#             row[no2_col] if no2_col else None,
#         )

#     return {
#         "temperature": float(row[temp_col]) if temp_col else None,
#         "humidity": float(row[humidity_col]) if humidity_col else None,
#         "pm25": float(row[pm25_col]) if pm25_col else None,
#         "current_aqi": current_aqi,
#         "_detected_columns": {
#             "temperature": temp_col, "humidity": humidity_col,
#             "pm25": pm25_col, "current_aqi": aqi_col,
#         },
#     }


# @st.cache_data(show_spinner=False)
# def load_evaluation_results() -> pd.DataFrame:
#     return pd.read_csv(EVALUATION_RESULTS)


# @st.cache_data(show_spinner=False)
# def load_shap_results(horizon: int) -> pd.DataFrame:
#     path = SHAP_DIRECTORY / f"shap_importance_{horizon}h.csv"
#     result = pd.read_csv(path)
#     return result.sort_values("mean_abs_shap", ascending=False)


# def render_aqi_gauge(value: float | None) -> go.Figure:
#     display_value = value if value is not None else 0
#     label, color = aqi_category(display_value)
#     figure = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=display_value,
#         number={"font": {"size": 42}, "suffix": " AQI"},
#         title={"text": f"<b>{label}</b>", "font": {"size": 18, "color": color}},
#         gauge={
#             "axis": {"range": [0, 500], "tickwidth": 1, "dtick": 100},
#             "bar": {"color": color, "thickness": 0.3},
#             "bgcolor": "#171a21",
#             "borderwidth": 0,
#             "steps": [
#                 {"range": [0, 50], "color": "#d9efe4"},
#                 {"range": [50, 100], "color": "#f1edc9"},
#                 {"range": [100, 150], "color": "#f4dec4"},
#                 {"range": [150, 200], "color": "#f2cccc"},
#                 {"range": [200, 300], "color": "#e6d5e9"},
#                 {"range": [300, 500], "color": "#e8cbc7"},
#             ],
#         },
#     ))
#     figure.update_layout(
#         template="plotly_dark",
#         height=310,
#         margin={"l": 20, "r": 20, "t": 55, "b": 10},
#         paper_bgcolor="#0f1117",
#         plot_bgcolor="#0f1117",
#         font={"color": "#f0f2fa"},
#     )
#     return figure


# @st.cache_data(show_spinner=False)
# def load_history(days: int = 7) -> pd.DataFrame:
#     fs = get_feature_store()
#     fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
#     df = fg.read()
#     df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
#     df = df.sort_values("timestamp").reset_index(drop=True)
#     cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
#     return df[df["timestamp"] >= cutoff].reset_index(drop=True)


# @st.cache_data(show_spinner=False)
# def load_forecast() -> tuple[pd.DataFrame, dict[str, Any]]:
#     latest_row, _fs = load_latest_row()
#     mr = get_model_registry()
#     predictions = predict_next_3_days(latest_row, mr)
#     conditions = get_current_conditions(latest_row)
#     return predictions, conditions


# st.title("🌫️ Karachi AQI Predictor")
# st.caption("Live 3-day forecast | 24h Ridge | 48h/72h XGBoost")

# with st.spinner("Loading latest conditions and forecast..."):
#     predictions, conditions = load_forecast()
#     alert_analysis = analyze_aqi_alerts(conditions["current_aqi"], predictions.to_dict("records"))

# overview_tab, evaluation_tab, shap_tab, eda_tab = st.tabs([
#     "Current AQI", "Model evaluation", "SHAP explainability", "EDA insights"
# ])

# with overview_tab:
#     gauge_col, conditions_col = st.columns([1, 1.5])
#     with gauge_col:
#         st.subheader("Current air quality")
#         st.plotly_chart(render_aqi_gauge(conditions["current_aqi"]), use_container_width=True)
#         st.caption("AQI scale: 0 = good air quality, 500 = hazardous.")

#     with conditions_col:
#         st.subheader("Live conditions")
#         metric_columns = st.columns(2)
#         values = [
#             ("Temperature", conditions["temperature"], "°C", ".1f"),
#             ("Relative humidity", conditions["humidity"], "%", ".0f"),
#             ("PM2.5", conditions["pm25"], " µg/m³", ".1f"),
#             ("Current AQI", conditions["current_aqi"], "", ".0f"),
#         ]
#         for column, (label, value, suffix, format_spec) in zip(metric_columns * 2, values):
#             with column:
#                 display_value = f"{value:{format_spec}}{suffix}" if value is not None else "N/A"
#                 st.metric(label, display_value)

#         latest_pollutants = load_history(days=1).tail(1)
#         pollutant_columns = ["pm2_5", "pm10", "o3", "no2", "so2", "co"]
#         available_pollutants = [c for c in pollutant_columns if c in latest_pollutants.columns]
#         if available_pollutants:
#             pollutant_frame = latest_pollutants[available_pollutants].T.rename(columns={latest_pollutants.index[-1]: "value"})
#             pollutant_frame["value"] = pd.to_numeric(pollutant_frame["value"], errors="coerce")
#             pollutant_frame = pollutant_frame.dropna().reset_index().rename(columns={"index": "pollutant"})
#             pollutant_fig = go.Figure(go.Bar(
#                 x=pollutant_frame["value"], y=pollutant_frame["pollutant"], orientation="h",
#                 marker_color="#55c2a5", text=pollutant_frame["value"].round(2), textposition="auto",
#             ))
#             pollutant_fig.update_layout(
#                 title="Pollutant levels",
#                 template="plotly_dark",
#                 height=260,
#                 margin={"l": 10, "r": 10, "t": 45, "b": 10},
#                 paper_bgcolor="#0f1117",
#                 plot_bgcolor="#0f1117",
#                 font={"color": "#f0f2fa"},
#                 xaxis={"gridcolor": "#343a46"},
#             )
#             st.plotly_chart(pollutant_fig, use_container_width=True)

#     if any(v is None for k, v in conditions.items() if k != "_detected_columns"):
#         with st.expander("Missing current-condition fields", expanded=True):
#             st.write("Some values could not be auto-detected from the feature group.")
#             st.json(conditions["_detected_columns"])

#     st.subheader("AQI alert analysis")
#     alert_level = alert_analysis["forecast_peak"]["alert_level"]
#     alert_colors = {"normal": "#2ecc71", "watch": "#f4d03f", "warning": "#e67e22", "critical": "#e74c3c", "unknown": "#8b93b0"}
#     alert_color = alert_colors.get(alert_level, "#8b93b0")
#     st.markdown(
#         f"""<div class="alert-card" style="border-color:{alert_color};">
#             <div class="title" style="color:{alert_color};">{alert_analysis['headline']}</div>
#             <div class="body">{alert_analysis['current']['advice']}</div>
#         </div>""", unsafe_allow_html=True,
#     )
#     with st.expander("Forecast alert details", expanded=True):
#         st.dataframe(pd.DataFrame(alert_analysis["forecast_rows"]), use_container_width=True, hide_index=True)

#     st.subheader("Forecast trajectory")
#     base_time = predictions["generated_at"].iloc[0]
#     x_values = [base_time] + predictions["forecast_for"].tolist()
#     y_values = [conditions["current_aqi"] if conditions["current_aqi"] is not None else predictions["predicted_aqi"].iloc[0]] + predictions["predicted_aqi"].tolist()
#     upper, lower = [y_values[0]], [y_values[0]]
#     for _, row in predictions.iterrows():
#         margin = 1.28 * row["holdout_rmse"] if pd.notna(row["holdout_rmse"]) else 0.0
#         upper.append(row["predicted_aqi"] + margin)
#         lower.append(max(0.0, row["predicted_aqi"] - margin))
#     forecast_fig = go.Figure()
#     forecast_fig.add_trace(go.Scatter(x=x_values + x_values[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(80,140,255,0.15)", line={"color": "rgba(0,0,0,0)"}, hoverinfo="skip", name="Approx. 80% interval"))
#     forecast_fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines+markers", line={"color": "#5b8dff", "width": 3}, name="Predicted AQI"))
#     forecast_fig.update_layout(
#         template="plotly_dark",
#         height=420,
#         xaxis_title="Time",
#         yaxis_title="AQI",
#         paper_bgcolor="#0f1117",
#         plot_bgcolor="#0f1117",
#         font={"color": "#f0f2fa"},
#     )
#     st.plotly_chart(forecast_fig, use_container_width=True)
#     with st.expander("Forecast details", expanded=True):
#         st.dataframe(predictions[["forecast_for", "horizon_hours", "predicted_aqi", "model_type", "holdout_rmse"]], use_container_width=True, hide_index=True)

#     st.subheader("Last 7 days")
#     history = load_history(days=7)
#     history_aqi_col = find_column(list(history.columns), ["us_aqi", "aqi"], exclude=["target", "predicted"])
#     if history_aqi_col is None:
#         st.info("Historical AQI data is not available in the feature group.")
#     else:
#         history_fig = go.Figure()
#         for low, high, _label, band_color in AQI_CATEGORIES:
#             history_fig.add_hrect(
#                 y0=low,
#                 y1=min(high, 300),
#                 fillcolor=band_color,
#                 opacity=0.10,
#                 line_width=0,
#             )
#         history_fig.add_trace(go.Scatter(
#             x=history["timestamp"],
#             y=history[history_aqi_col],
#             mode="lines+markers",
#             line={"color": "#f0f2fa", "width": 2},
#             marker={"color": "#f0f2fa", "size": 4},
#             name="Observed AQI",
#         ))
#         history_fig.update_layout(
#             template="plotly_dark",
#             height=380,
#             margin={"l": 10, "r": 10, "t": 10, "b": 10},
#             xaxis_title="Time",
#             yaxis_title="AQI",
#             yaxis={"gridcolor": "#343a46", "range": [0, 300]},
#             plot_bgcolor="#0f1117",
#             paper_bgcolor="#0f1117",
#             font={"color": "#f0f2fa"},
#             showlegend=False,
#         )
#         st.plotly_chart(history_fig, use_container_width=True)

# with evaluation_tab:
#     st.subheader("Why these champion models were selected")
#     metric = st.selectbox("Evaluation metric", ["rmse", "mae", "r2"], format_func=lambda value: value.upper(), index=0)
#     evaluation = load_evaluation_results().copy()
#     evaluation["model_label"] = evaluation["model"].str.replace("_pruned", "", regex=False)
#     evaluation["horizon_label"] = evaluation["horizon"].astype(str) + "h"
#     evaluation_fig = go.Figure()
#     for model in evaluation["model_label"].unique():
#         subset = evaluation[evaluation["model_label"] == model]
#         evaluation_fig.add_trace(go.Bar(x=subset["horizon_label"], y=subset[metric], name=model))
#     evaluation_fig.update_layout(
#         barmode="group",
#         template="plotly_dark",
#         height=430,
#         yaxis_title=metric.upper(),
#         xaxis_title="Forecast horizon",
#         plot_bgcolor="#0f1117",
#         paper_bgcolor="#0f1117",
#         font={"color": "#f0f2fa"},
#     )
#     st.plotly_chart(evaluation_fig, use_container_width=True)
#     st.caption("For RMSE and MAE, lower is better. For R², higher is better. The champion is selected independently for each horizon.")
#     winners = evaluation.loc[evaluation.groupby("horizon")[metric].idxmin() if metric != "r2" else evaluation.groupby("horizon")[metric].idxmax()].copy()
#     winners["Selection"] = winners.apply(lambda row: f"{row['model_label']} selected for {int(row['horizon'])}h", axis=1)
#     st.dataframe(winners[["horizon_label", "Selection", metric]], use_container_width=True, hide_index=True)
#     with st.expander("Full holdout metrics", expanded=True):
#         st.dataframe(evaluation[["model_label", "horizon_label", "rmse", "mae", "r2"]], use_container_width=True, hide_index=True)

# with shap_tab:
#     st.subheader("Global SHAP feature importance")
#     shap_horizon = st.selectbox("Forecast horizon", [24, 48, 72], format_func=lambda value: f"{value} hours", index=0)
#     shap_data = load_shap_results(shap_horizon).head(15).sort_values("mean_abs_shap")
#     shap_fig = go.Figure(go.Bar(x=shap_data["mean_abs_shap"], y=shap_data["feature"], orientation="h", marker_color="#55c2a5"))
#     shap_fig.update_layout(
#         template="plotly_dark",
#         height=560,
#         xaxis_title="Mean absolute SHAP value",
#         yaxis_title="Feature",
#         plot_bgcolor="#0f1117",
#         paper_bgcolor="#0f1117",
#         font={"color": "#f0f2fa"},
#     )
#     st.plotly_chart(shap_fig, use_container_width=True)
#     st.caption("Higher mean absolute SHAP values indicate greater average influence on the model output. They do not show whether a feature raises or lowers an individual forecast.")
#     with st.expander("Selected SHAP features", expanded=True):
#         st.dataframe(shap_data.sort_values("mean_abs_shap", ascending=False), use_container_width=True, hide_index=True)

# with eda_tab:
#     st.subheader("Exploratory data analysis insights")
#     st.caption("These views summarize the patterns used to guide feature engineering and model selection.")

#     eda_top_left, eda_top_right = st.columns(2)
#     with eda_top_left:
#         st.image(PROJECT_ROOT / "EDA-outputs" / "target_dist.png", caption="AQI target distribution", use_container_width=True)
#         st.markdown("**Continuous forecasting target**")
#         st.write("AQI spans a range of conditions, so the project predicts a continuous value before mapping it to health categories.")
#     with eda_top_right:
#         st.image(PROJECT_ROOT / "EDA-outputs" / "aqi_trend.png", caption="AQI trend over time", use_container_width=True)
#         st.markdown("**Temporal persistence**")
#         st.write("The time-series pattern supports lag and rolling features: recent AQI conditions contain useful information about the next forecast horizon.")

#     eda_bottom_left, eda_bottom_right = st.columns(2)
#     with eda_bottom_left:
#         st.image(PROJECT_ROOT / "EDA-outputs" / "feature_corr.png", caption="Feature correlation", use_container_width=True)
#         st.markdown("**Related environmental signals**")
#         st.write("Pollutant, weather, and engineered AQI variables are related, which is why the model uses regularization and horizon-specific feature pruning.")
#     with eda_bottom_right:
#         st.image(PROJECT_ROOT / "EDA-outputs" / "feature_dist.png", caption="Feature distributions", use_container_width=True)
#         st.markdown("**Different feature scales**")
#         st.write("Features have different ranges and distributions. Scaling, missing-value handling, and robust feature preparation are important parts of the training pipeline.")

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_RESULTS = PROJECT_ROOT / "results" / "shap_pruned_results.csv"
SHAP_DIRECTORY = PROJECT_ROOT / "SHAP"

# Styling
st.markdown(
    """
    <style>
    .stApp { background: #f7f8fa; color: #1a1d29; }
    [data-testid="stHeader"] { background: #f7f8fa; }
    [data-testid="stSidebar"] { background: #ffffff; }
    h1, h2, h3, p, label { color: #1a1d29; }
    .metric-card {
        background: #ffffff;
        border: 1px solid #dde1e8;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: left;
    }
    .metric-card .label { color: #6b7280; font-size: 0.8rem; margin-bottom: 4px; }
    .metric-card .value { color: #1a1d29; font-size: 1.8rem; font-weight: 600; line-height: 1.1; }
    .metric-card .sub { font-size: 0.8rem; margin-top: 4px; }
    .alert-card {
        border-radius: 12px;
        border: 1px solid;
        padding: 16px 18px;
        background: #ffffff;
    }
    .alert-card .title { font-size: 1.05rem; font-weight: 700; margin-bottom: 4px; }
    .alert-card .body { color: #333844; font-size: 0.95rem; }
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
def load_evaluation_results() -> pd.DataFrame:
    return pd.read_csv(EVALUATION_RESULTS)


@st.cache_data(show_spinner=False)
def load_shap_results(horizon: int) -> pd.DataFrame:
    path = SHAP_DIRECTORY / f"shap_importance_{horizon}h.csv"
    result = pd.read_csv(path)
    return result.sort_values("mean_abs_shap", ascending=False)


def render_aqi_gauge(value: float | None) -> go.Figure:
    display_value = value if value is not None else 0
    label, color = aqi_category(display_value)
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_value,
        number={"font": {"size": 42}, "suffix": " AQI"},
        title={"text": f"<b>{label}</b>", "font": {"size": 18, "color": color}},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1, "dtick": 100},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#ffffff",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#d9efe4"},
                {"range": [50, 100], "color": "#f1edc9"},
                {"range": [100, 150], "color": "#f4dec4"},
                {"range": [150, 200], "color": "#f2cccc"},
                {"range": [200, 300], "color": "#e6d5e9"},
                {"range": [300, 500], "color": "#e8cbc7"},
            ],
        },
    ))
    figure.update_layout(
        template="plotly_white",
        height=310,
        margin={"l": 20, "r": 20, "t": 55, "b": 10},
        paper_bgcolor="#f7f8fa",
        plot_bgcolor="#f7f8fa",
        font={"color": "#1a1d29"},
    )
    return figure


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
st.caption("Live 3-day forecast | 24h Ridge | 48h/72h XGBoost")

with st.spinner("Loading latest conditions and forecast..."):
    predictions, conditions = load_forecast()
    alert_analysis = analyze_aqi_alerts(conditions["current_aqi"], predictions.to_dict("records"))

overview_tab, evaluation_tab, shap_tab, eda_tab = st.tabs([
    "Current AQI", "Model evaluation", "SHAP explainability", "EDA insights"
])

with overview_tab:
    gauge_col, conditions_col = st.columns([1, 1.5])
    with gauge_col:
        st.subheader("Current air quality")
        st.plotly_chart(render_aqi_gauge(conditions["current_aqi"]), use_container_width=True)
        st.caption("AQI scale: 0 = good air quality, 500 = hazardous.")

    with conditions_col:
        st.subheader("Live conditions")
        metric_columns = st.columns(2)
        values = [
            ("Temperature", conditions["temperature"], "°C", ".1f"),
            ("Relative humidity", conditions["humidity"], "%", ".0f"),
            ("PM2.5", conditions["pm25"], " µg/m³", ".1f"),
            ("Current AQI", conditions["current_aqi"], "", ".0f"),
        ]
        for column, (label, value, suffix, format_spec) in zip(metric_columns * 2, values):
            with column:
                display_value = f"{value:{format_spec}}{suffix}" if value is not None else "N/A"
                st.metric(label, display_value)

        latest_pollutants = load_history(days=1).tail(1)
        pollutant_columns = ["pm2_5", "pm10", "o3", "no2", "so2", "co"]
        available_pollutants = [c for c in pollutant_columns if c in latest_pollutants.columns]
        if available_pollutants:
            pollutant_frame = latest_pollutants[available_pollutants].T.rename(columns={latest_pollutants.index[-1]: "value"})
            pollutant_frame["value"] = pd.to_numeric(pollutant_frame["value"], errors="coerce")
            pollutant_frame = pollutant_frame.dropna().reset_index().rename(columns={"index": "pollutant"})
            pollutant_fig = go.Figure(go.Bar(
                x=pollutant_frame["value"], y=pollutant_frame["pollutant"], orientation="h",
                marker_color="#2f9e78", text=pollutant_frame["value"].round(2), textposition="auto",
            ))
            pollutant_fig.update_layout(
                title="Pollutant levels",
                template="plotly_white",
                height=260,
                margin={"l": 10, "r": 10, "t": 45, "b": 10},
                paper_bgcolor="#f7f8fa",
                plot_bgcolor="#f7f8fa",
                font={"color": "#1a1d29"},
                xaxis={"gridcolor": "#dde1e8"},
            )
            st.plotly_chart(pollutant_fig, use_container_width=True)

    if any(v is None for k, v in conditions.items() if k != "_detected_columns"):
        with st.expander("Missing current-condition fields", expanded=True):
            st.write("Some values could not be auto-detected from the feature group.")
            st.json(conditions["_detected_columns"])

    st.subheader("AQI alert analysis")
    alert_level = alert_analysis["forecast_peak"]["alert_level"]
    alert_colors = {"normal": "#2ecc71", "watch": "#c9a600", "warning": "#e67e22", "critical": "#e74c3c", "unknown": "#6b7280"}
    alert_color = alert_colors.get(alert_level, "#6b7280")
    st.markdown(
        f"""<div class="alert-card" style="border-color:{alert_color};">
            <div class="title" style="color:{alert_color};">{alert_analysis['headline']}</div>
            <div class="body">{alert_analysis['current']['advice']}</div>
        </div>""", unsafe_allow_html=True,
    )
    with st.expander("Forecast alert details", expanded=True):
        st.dataframe(pd.DataFrame(alert_analysis["forecast_rows"]), use_container_width=True, hide_index=True)

    st.subheader("Forecast trajectory")
    base_time = predictions["generated_at"].iloc[0]
    x_values = [base_time] + predictions["forecast_for"].tolist()
    y_values = [conditions["current_aqi"] if conditions["current_aqi"] is not None else predictions["predicted_aqi"].iloc[0]] + predictions["predicted_aqi"].tolist()
    upper, lower = [y_values[0]], [y_values[0]]
    for _, row in predictions.iterrows():
        margin = 1.28 * row["holdout_rmse"] if pd.notna(row["holdout_rmse"]) else 0.0
        upper.append(row["predicted_aqi"] + margin)
        lower.append(max(0.0, row["predicted_aqi"] - margin))
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=x_values + x_values[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(80,140,255,0.15)", line={"color": "rgba(0,0,0,0)"}, hoverinfo="skip", name="Approx. 80% interval"))
    forecast_fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines+markers", line={"color": "#3b6fe0", "width": 3}, name="Predicted AQI"))
    forecast_fig.update_layout(
        template="plotly_white",
        height=420,
        xaxis_title="Time",
        yaxis_title="AQI",
        paper_bgcolor="#f7f8fa",
        plot_bgcolor="#f7f8fa",
        font={"color": "#1a1d29"},
    )
    st.plotly_chart(forecast_fig, use_container_width=True)
    with st.expander("Forecast details", expanded=True):
        st.dataframe(predictions[["forecast_for", "horizon_hours", "predicted_aqi", "model_type", "holdout_rmse"]], use_container_width=True, hide_index=True)

    st.subheader("Last 7 days")
    history = load_history(days=7)
    history_aqi_col = find_column(list(history.columns), ["us_aqi", "aqi"], exclude=["target", "predicted"])
    if history_aqi_col is None:
        st.info("Historical AQI data is not available in the feature group.")
    else:
        history_fig = go.Figure()
        for low, high, _label, band_color in AQI_CATEGORIES:
            history_fig.add_hrect(
                y0=low,
                y1=min(high, 300),
                fillcolor=band_color,
                opacity=0.10,
                line_width=0,
            )
        history_fig.add_trace(go.Scatter(
            x=history["timestamp"],
            y=history[history_aqi_col],
            mode="lines+markers",
            line={"color": "#1a1d29", "width": 2},
            marker={"color": "#1a1d29", "size": 4},
            name="Observed AQI",
        ))
        history_fig.update_layout(
            template="plotly_white",
            height=380,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            xaxis_title="Time",
            yaxis_title="AQI",
            yaxis={"gridcolor": "#dde1e8", "range": [0, 300]},
            plot_bgcolor="#f7f8fa",
            paper_bgcolor="#f7f8fa",
            font={"color": "#1a1d29"},
            showlegend=False,
        )
        st.plotly_chart(history_fig, use_container_width=True)

with evaluation_tab:
    st.subheader("Why these champion models were selected")
    metric = st.selectbox("Evaluation metric", ["rmse", "mae", "r2"], format_func=lambda value: value.upper(), index=0)
    evaluation = load_evaluation_results().copy()
    evaluation["model_label"] = evaluation["model"].str.replace("_pruned", "", regex=False)
    evaluation["horizon_label"] = evaluation["horizon"].astype(str) + "h"
    evaluation_fig = go.Figure()
    for model in evaluation["model_label"].unique():
        subset = evaluation[evaluation["model_label"] == model]
        evaluation_fig.add_trace(go.Bar(x=subset["horizon_label"], y=subset[metric], name=model))
    evaluation_fig.update_layout(
        barmode="group",
        template="plotly_white",
        height=430,
        yaxis_title=metric.upper(),
        xaxis_title="Forecast horizon",
        plot_bgcolor="#f7f8fa",
        paper_bgcolor="#f7f8fa",
        font={"color": "#1a1d29"},
    )
    st.plotly_chart(evaluation_fig, use_container_width=True)
    st.caption("For RMSE and MAE, lower is better. For R², higher is better. The champion is selected independently for each horizon.")
    winners = evaluation.loc[evaluation.groupby("horizon")[metric].idxmin() if metric != "r2" else evaluation.groupby("horizon")[metric].idxmax()].copy()
    winners["Selection"] = winners.apply(lambda row: f"{row['model_label']} selected for {int(row['horizon'])}h", axis=1)
    st.dataframe(winners[["horizon_label", "Selection", metric]], use_container_width=True, hide_index=True)
    with st.expander("Full holdout metrics", expanded=True):
        st.dataframe(evaluation[["model_label", "horizon_label", "rmse", "mae", "r2"]], use_container_width=True, hide_index=True)

with shap_tab:
    st.subheader("Global SHAP feature importance")
    shap_horizon = st.selectbox("Forecast horizon", [24, 48, 72], format_func=lambda value: f"{value} hours", index=0)
    shap_data = load_shap_results(shap_horizon).head(15).sort_values("mean_abs_shap")
    shap_fig = go.Figure(go.Bar(x=shap_data["mean_abs_shap"], y=shap_data["feature"], orientation="h", marker_color="#2f9e78"))
    shap_fig.update_layout(
        template="plotly_white",
        height=560,
        xaxis_title="Mean absolute SHAP value",
        yaxis_title="Feature",
        plot_bgcolor="#f7f8fa",
        paper_bgcolor="#f7f8fa",
        font={"color": "#1a1d29"},
    )
    st.plotly_chart(shap_fig, use_container_width=True)
    st.caption("Higher mean absolute SHAP values indicate greater average influence on the model output. They do not show whether a feature raises or lowers an individual forecast.")
    with st.expander("Selected SHAP features", expanded=True):
        st.dataframe(shap_data.sort_values("mean_abs_shap", ascending=False), use_container_width=True, hide_index=True)

with eda_tab:
    st.subheader("Exploratory data analysis insights")
    st.caption("These views summarize the patterns used to guide feature engineering and model selection.")

    eda_top_left, eda_top_right = st.columns(2)
    with eda_top_left:
        st.image(PROJECT_ROOT / "EDA-outputs" / "target_dist.png", caption="AQI target distribution", use_container_width=True)
        st.markdown("**Continuous forecasting target**")
        st.write("AQI spans a range of conditions, so the project predicts a continuous value before mapping it to health categories.")
    with eda_top_right:
        st.image(PROJECT_ROOT / "EDA-outputs" / "aqi_trend.png", caption="AQI trend over time", use_container_width=True)
        st.markdown("**Temporal persistence**")
        st.write("The time-series pattern supports lag and rolling features: recent AQI conditions contain useful information about the next forecast horizon.")

    eda_bottom_left, eda_bottom_right = st.columns(2)
    with eda_bottom_left:
        st.image(PROJECT_ROOT / "EDA-outputs" / "feature_corr.png", caption="Feature correlation", use_container_width=True)
        st.markdown("**Related environmental signals**")
        st.write("Pollutant, weather, and engineered AQI variables are related, which is why the model uses regularization and horizon-specific feature pruning.")
    with eda_bottom_right:
        st.image(PROJECT_ROOT / "EDA-outputs" / "feature_dist.png", caption="Feature distributions", use_container_width=True)
        st.markdown("**Different feature scales**")
        st.write("Features have different ranges and distributions. Scaling, missing-value handling, and robust feature preparation are important parts of the training pipeline.")