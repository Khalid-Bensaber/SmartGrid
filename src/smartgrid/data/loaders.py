from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from smartgrid.common.constants import (
    DEFAULT_AIRTEMP_VALUE,
    DEFAULT_TARGET_NAME,
    OLD_FORECAST_COLUMNS,
    TOTAL_COLUMNS,
)

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


def load_holiday_sets(holidays_xlsx: str | Path) -> tuple[set, set]:
    xls = pd.ExcelFile(holidays_xlsx)
    holiday_dates: set = set()
    special_dates: set = set()

    for sheet in xls.sheet_names:
        df = pd.read_excel(holidays_xlsx, sheet_name=sheet)
        if "Unnamed: 0" in df.columns:
            s = pd.to_datetime(df["Unnamed: 0"], errors="coerce").dropna().dt.date
            holiday_dates.update(s.tolist())
        if "Unnamed: 2" in df.columns:
            s2 = pd.to_datetime(df["Unnamed: 2"], errors="coerce").dropna().dt.date
            special_dates.update(s2.tolist())

    return holiday_dates, special_dates


def load_history(csv_path: str | Path, date_col: str = "Date") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    missing_cols = [c for c in TOTAL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required history columns: {missing_cols}")

    df[DEFAULT_TARGET_NAME] = df[TOTAL_COLUMNS].sum(axis=1)

    if "Airtemp" not in df.columns:
        if "AirTemp" in df.columns:
            df["Airtemp"] = df["AirTemp"]
        else:
            df["Airtemp"] = DEFAULT_AIRTEMP_VALUE

    return df


def load_weather_history(weather_csv: str | Path | None, date_col: str = "Date") -> pd.DataFrame | None:
    if weather_csv is None:
        return None

    path = Path(weather_csv)
    if not path.exists():
        warnings.warn(f"Weather CSV not found: {weather_csv}")
        return None

    weather = pd.read_csv(path)
    weather[date_col] = pd.to_datetime(weather[date_col], errors="coerce")
    weather = weather.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    keep_cols = [date_col] + [c for c in WEATHER_RAW_COLUMNS if c in weather.columns]
    missing = [c for c in WEATHER_RAW_COLUMNS if c not in weather.columns]
    if missing:
        warnings.warn(f"Weather CSV missing optional columns: {missing}")

    weather = weather[keep_cols].copy()
    weather = weather.rename(columns=WEATHER_RENAME_MAP)
    weather = weather.drop_duplicates(subset=[date_col], keep="last").reset_index(drop=True)
    return weather


def merge_weather_on_history(hist: pd.DataFrame, weather: pd.DataFrame | None, date_col: str = "Date") -> pd.DataFrame:
    if weather is None:
        return hist

    merged = hist.merge(weather, on=date_col, how="left")
    weather_cols = [c for c in merged.columns if c.startswith("Weather_")]

    if weather_cols:
        merged = merged.sort_values(date_col).reset_index(drop=True)
        merged[weather_cols] = merged[weather_cols].interpolate(method="linear", limit_direction="both")
        merged[weather_cols] = merged[weather_cols].ffill().bfill()

    return merged


def load_old_benchmark(benchmark_csv: str | Path | None, date_col: str = "Date") -> pd.DataFrame | None:
    if benchmark_csv is None:
        return None
    path = Path(benchmark_csv)
    if not path.exists():
        warnings.warn(f"Benchmark CSV not found: {benchmark_csv}")
        return None

    old = pd.read_csv(path)
    old[date_col] = pd.to_datetime(old[date_col], errors="coerce")
    old = old.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    missing = [c for c in OLD_FORECAST_COLUMNS if c not in old.columns]
    if missing:
        warnings.warn(f"Benchmark CSV missing columns needed for notebook total: {missing}")
        return None

    old["OldLegacy_TOTAL_Forecast"] = old[OLD_FORECAST_COLUMNS].sum(axis=1)
    keep_cols = [date_col, "OldLegacy_TOTAL_Forecast"]
    if "Ptot_Ilot_Forecast" in old.columns:
        keep_cols.append("Ptot_Ilot_Forecast")
    return old[keep_cols].copy()