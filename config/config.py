"""
Central configuration for Pearls AQI Predictor.
"""

# ========================================================================
# Location — Karachi centroid
# ========================================================================

KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011

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