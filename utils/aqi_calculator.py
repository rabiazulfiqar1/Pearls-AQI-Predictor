"""
AQI calculation and alert analysis helpers.

Uses the US EPA piecewise linear interpolation formula:

    AQI = ((I_hi - I_lo) / (C_hi - C_lo)) * (C - C_lo) + I_lo

where (C_lo, C_hi) is the concentration breakpoint bracketing C, and
(I_lo, I_hi) is the corresponding AQI breakpoint.

The overall AQI at a timestamp is the MAXIMUM of the sub-indices across
all pollutants (US EPA "dominant pollutant" rule).

Units expected from Open-Meteo:
    pm2_5, pm10, so2, no2, o3   - ug/m3
    co                           - ug/m3 (converted to mg/m3 internally)

Averaging windows per EPA:
    pm2_5   - 24h
    pm10    - 24h
    o3      - 8h
    co      - 8h
    so2     - 1h
    no2     - 1h

The feature pipeline is responsible for supplying already-averaged values.
This module applies the breakpoint lookup only.

NOTE: as of the current feature pipeline, compute_aqi() is NOT used to
populate the `aqi` training column -- that column is Open-Meteo's own
us_aqi field, taken directly, since it's already derived from the same
CAMS pollutant data via this same formula. compute_aqi() is kept here as
a documented fallback for any future data source that only reports raw
pollutant concentrations. get_category() and get_category_label() ARE
still used regardless of where the numeric AQI comes from.
"""
import math
from collections.abc import Iterable
from typing import Any

from config.config import AQI_CATEGORIES

# ========================================================================
# EPA breakpoints
# Each entry: (C_lo, C_hi, I_lo, I_hi)
# Concentrations in ug/m3 except CO which is in mg/m3
# Source: US EPA Technical Assistance Document (May 2024)
# ========================================================================

BREAKPOINTS = {
    "pm2_5": [
        (0.0,   9.0,   0,   50),
        (9.1,   35.4,  51,  100),
        (35.5,  55.4,  101, 150),
        (55.5,  125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
    ],
    "pm10": [
        (0,   54,   0,   50),
        (55,  154,  51,  100),
        (155, 254,  101, 150),
        (255, 354,  151, 200),
        (355, 424,  201, 300),
        (425, 604,  301, 500),
    ],
    "o3": [
        (0,   108, 0,   50),
        (109, 140, 51,  100),
        (141, 170, 101, 150),
        (171, 210, 151, 200),
        (211, 400, 201, 300),
    ],
    "co": [
        (0.0,  5.0,  0,   50),
        (5.1,  10.5, 51,  100),
        (10.6, 14.0, 101, 150),
        (14.1, 17.5, 151, 200),
        (17.6, 34.9, 201, 300),
        (35.0, 57.5, 301, 500),
    ],
    "so2": [
        (0,   92,   0,   50),
        (93,  197,  51,  100),
        (198, 484,  101, 150),
        (485, 796,  151, 200),
        (797, 1583, 201, 300),
    ],
    "no2": [
        (0,    100,  0,   50),
        (101,  188,  51,  100),
        (189,  677,  101, 150),
        (678,  1220, 151, 200),
        (1221, 2349, 201, 300),
    ],
}


def _sub_index(concentration: float, breakpoints: list) -> float | None:
    """
    Compute the AQI sub-index for a single pollutant via piecewise linear
    interpolation against its breakpoint table.
    """
    if concentration is None:
        return None
    try:
        c = float(concentration)
    except (TypeError, ValueError):
        return None
    if math.isnan(c):            # NaN check
        return None
    if c < 0:
        return None

    n = len(breakpoints)
    for i, (c_lo, c_hi, i_lo, i_hi) in enumerate(breakpoints):
        upper = breakpoints[i + 1][0] if i + 1 < n else c_hi
        is_final = i + 1 == n
        in_band = (c_lo <= c <= upper) if is_final else (c_lo <= c < upper)
        if in_band:
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (c - c_lo) + i_lo

    return 500.0


def compute_aqi(
    pm2_5: float | None,
    pm10:  float | None,
    o3:    float | None,
    co:    float | None,
    so2:   float | None,
    no2:   float | None,
) -> float | None:
    """
    Compute the overall AQI as the maximum sub-index across available
    pollutants (US EPA dominant pollutant rule).

    Not currently called by the feature pipeline (see module docstring) --
    kept as a fallback for data sources that don't provide a ready-made
    AQI field.
    """
    co_mg = None if co is None else co / 1000.0

    sub_indices = [
        _sub_index(pm2_5, BREAKPOINTS["pm2_5"]),
        _sub_index(pm10,  BREAKPOINTS["pm10"]),
        _sub_index(o3,    BREAKPOINTS["o3"]),
        _sub_index(co_mg, BREAKPOINTS["co"]),
        _sub_index(so2,   BREAKPOINTS["so2"]),
        _sub_index(no2,   BREAKPOINTS["no2"]),
    ]

    valid = [s for s in sub_indices if s is not None]
    if not valid:
        return None

    return round(max(valid), 2)


def get_category(aqi: float | None) -> int | None:
    """Return the AQI category ID (1-6) for a given AQI value."""
    if aqi is None:
        return None
    try:
        aqi_val = float(aqi)
    except (TypeError, ValueError):
        return None
    if math.isnan(aqi_val):
        return None

    for (lo, hi), (cat_id, _label) in AQI_CATEGORIES.items():
        if lo <= aqi_val <= hi:
            return cat_id

    if aqi_val > 500:
        return 6
    return None


def get_category_label(aqi: float | None) ->str | None:
    """Return the AQI category label (e.g. 'Moderate') for a given AQI."""
    if aqi is None:
        return None
    try:
        aqi_val = float(aqi)
    except (TypeError, ValueError):
        return None
    if math.isnan(aqi_val):
        return None

    for (lo, hi), (_cat_id, label) in AQI_CATEGORIES.items():
        if lo <= aqi_val <= hi:
            return label

    if aqi_val > 500:
        return "Hazardous"
    return None


def get_aqi_alert_level(aqi: float | None) -> str:
    """Map an AQI value to a simple alert level string."""
    category = get_category(aqi)
    if category is None:
        return "unknown"
    if category <= 2:
        return "normal"
    if category == 3:
        return "watch"
    if category == 4:
        return "warning"
    return "critical"


def get_aqi_health_advice(aqi: float | None) -> str:
    """Return a short, user-facing action summary for an AQI value."""
    category = get_category(aqi)
    if category is None:
        return "AQI is unavailable, so no alert advice can be generated yet."
    if category == 1:
        return "Air quality is good. Normal outdoor activity is fine."
    if category == 2:
        return "Air quality is acceptable. Sensitive people should monitor symptoms."
    if category == 3:
        return "Sensitive groups should reduce prolonged or heavy outdoor exertion."
    if category == 4:
        return "Reduce outdoor exertion and keep sensitive groups indoors if possible."
    if category == 5:
        return "Avoid extended outdoor activity and consider rescheduling strenuous plans."
    return "Everyone should avoid outdoor exposure and follow local guidance."


def analyze_aqi_alerts(
    current_aqi: float | None,
    forecast_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize AQI risk across the current reading and forecast values."""
    forecast_rows = list(forecast_rows)
    forecast_values: list[tuple[float, dict[str, Any]]] = []
    for row in forecast_rows:
        value = row.get("predicted_aqi")
        try:
            value = None if value is None else float(value)
        except (TypeError, ValueError):
            value = None
        if value is not None and not math.isnan(value):
            forecast_values.append((value, row))

    current_label = get_category_label(current_aqi)
    current_level = get_aqi_alert_level(current_aqi)
    current_advice = get_aqi_health_advice(current_aqi)

    if forecast_values:
        peak_aqi, peak_row = max(forecast_values, key=lambda item: item[0])
        peak_label = get_category_label(peak_aqi)
        peak_level = get_aqi_alert_level(peak_aqi)
        peak_horizon = peak_row.get("horizon_hours")
        peak_forecast_for = peak_row.get("forecast_for")
    else:
        peak_aqi = None
        peak_label = None
        peak_level = "unknown"
        peak_horizon = None
        peak_forecast_for = None

    alert_values = [
        value for value, _ in forecast_values
        if get_aqi_alert_level(value) in {"watch", "warning", "critical"}
    ]
    has_alert = current_level in {"watch", "warning", "critical"} or bool(alert_values)

    if current_level == "critical" or peak_level == "critical":
        headline = "Critical AQI alert"
    elif current_level == "warning" or peak_level == "warning":
        headline = "AQI warning"
    elif current_level == "watch" or peak_level == "watch":
        headline = "AQI watch"
    else:
        headline = "AQI looks manageable"

    return {
        "headline": headline,
        "has_alert": has_alert,
        "current": {
            "aqi": current_aqi,
            "label": current_label,
            "alert_level": current_level,
            "advice": current_advice,
        },
        "forecast_peak": {
            "aqi": peak_aqi,
            "label": peak_label,
            "alert_level": peak_level,
            "horizon_hours": peak_horizon,
            "forecast_for": peak_forecast_for,
        },
        "forecast_rows": [
            {
                **row,
                "aqi_label": get_category_label(row.get("predicted_aqi")),
                "alert_level": get_aqi_alert_level(row.get("predicted_aqi")),
            }
            for row in forecast_rows
        ],
    }
