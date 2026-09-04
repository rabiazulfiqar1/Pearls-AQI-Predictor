from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    st._config.set_option("theme.base", "light")
    st._config.set_option("theme.backgroundColor", "#f7f9fc")
    st._config.set_option("theme.secondaryBackgroundColor", "#ffffff")
    st._config.set_option("theme.textColor", "#1a1d29")
    st._config.set_option("theme.primaryColor", "#0f766e")
    st._config.set_option("theme.font", "sans serif")
except Exception:
    # Private API — if it ever breaks across Streamlit versions, fall back to
    # a .streamlit/config.toml with the same [theme] values instead.
    pass

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

# ---------------------------------------------------------------------------
# Single consistent color theme (used everywhere: page, cards, and charts)
# ---------------------------------------------------------------------------
BG_PAGE = "#f7f9fc"        # page / chart background
BG_CARD = "#ffffff"        # card / sidebar background
BORDER = "#e1e5ee"         # card + gridline borders
TEXT_PRIMARY = "#1a1d29"   # body text
TEXT_SECONDARY = "#6b7280" # captions / muted labels
ACCENT = "#0f766e"         # single accent color for all headings
GRIDLINE = "#c7cedb"       # darker, clearly visible gridlines on the light bg
LINE_PRIMARY = "#1d4ed8"   # strong blue for the main forecast/observed line
LINE_BAND = "#93a5c9"      # muted blue-gray for the confidence band outline
PLOTLY_TEMPLATE = "plotly_white"

# AQI category thresholds (US EPA) — used for coloring + band shading
AQI_CATEGORIES = [
    (0, 50, "Good", "#2ecc71"),
    (50, 100, "Moderate", "#f4d03f"),
    (100, 150, "Unhealthy (Sensitive)", "#e67e22"),
    (150, 200, "Unhealthy", "#e74c3c"),
    (200, 300, "Very Unhealthy", "#9b59b6"),
    (300, 500, "Hazardous", "#7b241c"),
]

ALERT_COLORS = {
    "normal": "#2ecc71",
    "watch": "#c9a600",
    "warning": "#e67e22",
    "critical": "#e74c3c",
    "unknown": TEXT_SECONDARY,
}

st.markdown(
    f"""
    <style>
    .stApp {{ background: {BG_PAGE}; color: {TEXT_PRIMARY}; }}
    [data-testid="stHeader"] {{ background: {BG_PAGE}; }}
    [data-testid="stSidebar"] {{ background: {BG_CARD}; }}
    p, label, span, div {{ color: {TEXT_PRIMARY}; }}
    h1, h2, h3 {{ color: {ACCENT} !important; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {TEXT_SECONDARY} !important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 2px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        height: 44px;
        padding: 0 18px;
        border: 1px solid {BORDER};
        border-bottom: 0;
        border-radius: 8px 8px 0 0;
        background: #e9eef6;
        color: #526174 !important;
        font-weight: 700;
    }}
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {{ color: #526174 !important; font-weight: 700; }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: {ACCENT};
        border-color: {ACCENT};
        color: #ffffff !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span {{ color: #ffffff !important; }}
    div[data-baseweb="select"] > div {{
        background: #e9eef6;
        border: 2px solid {ACCENT};
        border-radius: 7px;
        color: {TEXT_PRIMARY};
    }}
    div[data-baseweb="select"] span {{ color: {TEXT_PRIMARY} !important; font-weight: 700; }}
    div[data-baseweb="popover"] {{ background: {BG_CARD}; border: 1px solid {BORDER}; }}
    div[role="listbox"] {{ background: {BG_CARD}; }}
    div[role="option"] {{ color: {TEXT_PRIMARY} !important; background: {BG_CARD}; }}
    div[role="option"][aria-selected="true"] {{ background: #dce9f8; color: {ACCENT} !important; font-weight: 700; }}
    .metric-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 16px 20px;
        text-align: left;
    }}
    .metric-card .label {{ color: {TEXT_SECONDARY}; font-size: 0.8rem; margin-bottom: 4px; }}
    .metric-card .value {{ color: {TEXT_PRIMARY}; font-size: 1.8rem; font-weight: 600; line-height: 1.1; }}
    .metric-card .sub {{ font-size: 0.8rem; margin-top: 4px; }}
    .alert-card {{
        border-radius: 12px;
        border: 1px solid;
        padding: 16px 18px;
        background: {BG_CARD};
    }}
    .alert-card .title {{ font-size: 1.05rem; font-weight: 700; margin-bottom: 4px; }}
    .alert-card .body {{ color: {TEXT_PRIMARY}; font-size: 0.95rem; }}
    [data-testid="stMetricValue"] {{ color: {TEXT_PRIMARY}; }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_SECONDARY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


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


def style_table(frame: pd.DataFrame):
    return frame.style.set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", ACCENT),
                ("color", "#ffffff"),
                ("font-weight", "700"),
                ("font-size", "0.9rem"),
                ("padding", "10px 12px"),
                ("text-align", "left"),
                ("border", f"1px solid {ACCENT}"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("border", f"1px solid {BORDER}"),
                ("padding", "8px 12px"),
                ("color", TEXT_PRIMARY),
            ],
        },
    ])


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


def _apply_chart_theme(figure: go.Figure, **layout_kwargs: Any) -> go.Figure:
    """Apply the single shared theme to every Plotly figure in the app.

    Any caller-supplied ``xaxis``/``yaxis`` dicts are merged on top of the
    default axis styling instead of overwriting it, so gridlines/tick colors
    stay consistent everywhere while callers can still set things like
    ``range``.
    """
    default_axis = {
        "gridcolor": GRIDLINE,
        "zerolinecolor": GRIDLINE,
        "linecolor": GRIDLINE,
        "tickfont": {"color": TEXT_PRIMARY},
        "title": {"font": {"color": TEXT_PRIMARY}},
    }
    xaxis = {**default_axis, **layout_kwargs.pop("xaxis", {})}
    yaxis = {**default_axis, **layout_kwargs.pop("yaxis", {})}
    legend = {"font": {"color": TEXT_PRIMARY}, **layout_kwargs.pop("legend", {})}

    figure.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG_PAGE,
        plot_bgcolor=BG_CARD,
        font={"color": TEXT_PRIMARY, "size": 13},
        xaxis=xaxis,
        yaxis=yaxis,
        legend=legend,
        **layout_kwargs,
    )
    return figure


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
            "bgcolor": BG_CARD,
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
    return _apply_chart_theme(figure, height=310, margin={"l": 20, "r": 20, "t": 55, "b": 10})


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
                marker_color=ACCENT, text=pollutant_frame["value"].round(2), textposition="auto",
            ))
            _apply_chart_theme(
                pollutant_fig,
                title="Pollutant levels",
                height=260,
                margin={"l": 10, "r": 10, "t": 45, "b": 10},
            )
            st.plotly_chart(pollutant_fig, use_container_width=True)

    if any(v is None for k, v in conditions.items() if k != "_detected_columns"):
        with st.expander("Missing current-condition fields", expanded=True):
            st.write("Some values could not be auto-detected from the feature group.")
            st.json(conditions["_detected_columns"])

    st.subheader("AQI alert analysis")
    alert_level = alert_analysis["forecast_peak"]["alert_level"]
    alert_color = ALERT_COLORS.get(alert_level, TEXT_SECONDARY)
    st.markdown(
        f"""<div class="alert-card" style="border-color:{alert_color};">
            <div class="title" style="color:{alert_color};">{alert_analysis['headline']}</div>
            <div class="body">{alert_analysis['current']['advice']}</div>
        </div>""", unsafe_allow_html=True,
    )
    with st.expander("Forecast alert details", expanded=True):
        st.dataframe(style_table(pd.DataFrame(alert_analysis["forecast_rows"])), use_container_width=True, hide_index=True)

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
    forecast_fig.add_trace(go.Scatter(
        x=x_values + x_values[::-1], y=upper + lower[::-1], fill="toself",
        fillcolor="rgba(29,78,216,0.14)", line={"color": LINE_BAND, "width": 1},
        hoverinfo="skip", name="Approx. 80% interval",
    ))
    forecast_fig.add_trace(go.Scatter(
        x=x_values, y=y_values, mode="lines+markers",
        line={"color": LINE_PRIMARY, "width": 3},
        marker={"color": LINE_PRIMARY, "size": 7, "line": {"color": BG_CARD, "width": 1}},
        name="Predicted AQI",
    ))
    _apply_chart_theme(
        forecast_fig, height=420, xaxis_title="Time", yaxis_title="AQI",
        legend={"orientation": "h", "y": 1.1, "font": {"color": TEXT_PRIMARY}},
    )
    st.plotly_chart(forecast_fig, use_container_width=True)
    with st.expander("Forecast details", expanded=True):
        st.dataframe(style_table(predictions[["forecast_for", "horizon_hours", "predicted_aqi", "model_type", "holdout_rmse"]]), use_container_width=True, hide_index=True)

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
                opacity=0.22,
                line_width=0,
                layer="below",
            )
        history_fig.add_trace(go.Scatter(
            x=history["timestamp"],
            y=history[history_aqi_col],
            mode="lines+markers",
            line={"color": LINE_PRIMARY, "width": 3},
            marker={"color": LINE_PRIMARY, "size": 5, "line": {"color": BG_CARD, "width": 1}},
            name="Observed AQI",
        ))
        _apply_chart_theme(
            history_fig,
            height=380,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            xaxis_title="Time",
            yaxis_title="AQI",
            yaxis={"range": [0, 300]},
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
    _apply_chart_theme(
        evaluation_fig,
        barmode="group",
        height=430,
        yaxis_title=metric.upper(),
        xaxis_title="Forecast horizon",
    )
    st.plotly_chart(evaluation_fig, use_container_width=True)
    st.caption("For RMSE and MAE, lower is better. For R², higher is better. The champion is selected independently for each horizon.")
    winners = evaluation.loc[evaluation.groupby("horizon")[metric].idxmin() if metric != "r2" else evaluation.groupby("horizon")[metric].idxmax()].copy()
    winners["Selection"] = winners.apply(lambda row: f"{row['model_label']} selected for {int(row['horizon'])}h", axis=1)
    st.dataframe(style_table(winners[["horizon_label", "Selection", metric]]), use_container_width=True, hide_index=True)
    with st.expander("Full holdout metrics", expanded=True):
        st.dataframe(style_table(evaluation[["model_label", "horizon_label", "rmse", "mae", "r2"]]), use_container_width=True, hide_index=True)

with shap_tab:
    st.subheader("Global SHAP feature importance")
    shap_horizon = st.selectbox("Forecast horizon", [24, 48, 72], format_func=lambda value: f"{value} hours", index=0)
    shap_data = load_shap_results(shap_horizon).head(15).sort_values("mean_abs_shap")
    shap_fig = go.Figure(go.Bar(x=shap_data["mean_abs_shap"], y=shap_data["feature"], orientation="h", marker_color=ACCENT))
    _apply_chart_theme(shap_fig, height=560, xaxis_title="Mean absolute SHAP value", yaxis_title="Feature")
    st.plotly_chart(shap_fig, use_container_width=True)
    st.caption("Higher mean absolute SHAP values indicate greater average influence on the model output. They do not show whether a feature raises or lowers an individual forecast.")
    with st.expander("Selected SHAP features", expanded=True):
        st.dataframe(style_table(shap_data.sort_values("mean_abs_shap", ascending=False)), use_container_width=True, hide_index=True)

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