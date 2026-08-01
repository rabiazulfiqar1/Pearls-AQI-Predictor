"""
Central configuration for Pearls AQI Predictor.

Single source of truth for all constants. Every pipeline imports from here.
Do NOT hardcode values in other files — add them here and import.
"""
# ========================================================================
# API Configuration
# ========================================================================

# Coordinates for Karachi data retrieval
KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

# ========================================================================
# Hopsworks feature group
# ========================================================================
FEATURE_GROUP_NAME    = "aqi_features"
FEATURE_GROUP_VERSION = 1

# Retry configuration for failed API requests
API_RETRY_COUNT = 3
API_RETRY_BASE_DELAY = 5

# ========================================================================
# Feature engineering — lag features (hours)
# ========================================================================
# Aligned with TARGET_HORIZONS_HOURS below by design: lagging AQI/pollutants
# by the same depths we forecast ahead gives a clean "same distance back as
# we're predicting forward" symmetry (e.g. the 24h lag is directly
# comparable to the 24h-ahead target).
LAG_HOURS = [24, 48, 72]

# ========================================================================
# Feature engineering — rolling windows (hours)
# ========================================================================
ROLLING_WINDOW_SHORT    = 6
ROLLING_WINDOW_LONG     = 24
ROLLING_WINDOW_BASELINE = 720   # 30 days

# ========================================================================
# Feature engineering — AQI change-rate windows (hours)
# ========================================================================
CHANGE_RATE_WINDOWS_HOURS = [1, 24]

# ========================================================================
# Forecast targets — horizons ahead (hours)
# ========================================================================
TARGET_HORIZONS_HOURS = [24, 48, 72]

# ========================================================================
# AQI categories (US EPA standard)
# Range (inclusive low, inclusive high) → (category_id, label)
# ========================================================================
AQI_CATEGORIES = {
    (0,   50):  (1, "Good"),
    (51,  100): (2, "Moderate"),
    (101, 150): (3, "Unhealthy for Sensitive Groups"),
    (151, 200): (4, "Unhealthy"),
    (201, 300): (5, "Very Unhealthy"),
    (301, 500): (6, "Hazardous"),
}