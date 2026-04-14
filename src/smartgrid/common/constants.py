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

WEATHER_RAW_COLUMNS = [
    "AirTemp",
    "CloudOpacity",
    "Dni10",
    "Dni90",
    "DniMoy",
    "Ghi10",
    "Ghi90",
    "GhiMoy",
]

WEATHER_RENAME_MAP = {
    "AirTemp": "Weather_AirTemp",
    "CloudOpacity": "Weather_CloudOpacity",
    "Dni10": "Weather_Dni10",
    "Dni90": "Weather_Dni90",
    "DniMoy": "Weather_DniMoy",
    "Ghi10": "Weather_Ghi10",
    "Ghi90": "Weather_Ghi90",
    "GhiMoy": "Weather_GhiMoy",
}

DEFAULT_WEATHER_COLUMNS = [
    "Weather_AirTemp",
    "Weather_CloudOpacity",
    "Weather_Dni10",
    "Weather_Dni90",
    "Weather_DniMoy",
    "Weather_Ghi10",
    "Weather_Ghi90",
    "Weather_GhiMoy",
]

BASIC_WEATHER_COLUMNS = [
    "Weather_AirTemp",
    "Weather_CloudOpacity",
]

IRRADIANCE_WEATHER_COLUMNS = [
    "Weather_Dni10",
    "Weather_Dni90",
    "Weather_DniMoy",
    "Weather_Ghi10",
    "Weather_Ghi90",
    "Weather_GhiMoy",
]

DEFAULT_TARGET_NAME = "tot"
DEFAULT_AIRTEMP_VALUE = 15.0
N_STEPS_PER_DAY = 144
TOL_REL = 0.01
TOL_ABS = 5000.0
