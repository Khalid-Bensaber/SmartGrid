from __future__ import annotations

TOTAL_COLUMNS = ["Ptot_HA", "Ptot_HEI_13RT", "Ptot_HEI_5RNS", "Ptot_RIZOMM"]
OLD_FORECAST_COLUMNS = [
    "Ptot_HA_Forecast",
    "Ptot_HEI_13RT_Forecast",
    "Ptot_HEI_5RNS_Forecast",
    "Ptot_RIZOMM_Forecast",
]
FORECAST_SCHEMA_COLUMNS = [
    "name",
    "Date",
    "Ptot_HA_Forecast",
    "Ptot_HEI_13RT_Forecast",
    "Ptot_HEI_5RNS_Forecast",
    "Ptot_Ilot_Forecast",
    "Ptot_RIZOMM_Forecast",
]

DEFAULT_TARGET_NAME = "tot"
DEFAULT_AIRTEMP_VALUE = 15.0
N_STEPS_PER_DAY = 144
TOL_REL = 0.01
TOL_ABS = 5000.0
