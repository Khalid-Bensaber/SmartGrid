from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from smartgrid.common.constants import (
    DEFAULT_AIRTEMP_VALUE,
    DEFAULT_TARGET_NAME,
    FORECAST_FREQ,
    N_STEPS_PER_DAY,
    OLD_FORECAST_COLUMNS,
    TOTAL_COLUMNS,
    WEATHER_RAW_COLUMNS,
    WEATHER_RENAME_MAP,
)
from smartgrid.data.timeline import build_timeline_diagnostics, sort_and_validate_timestamps


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


def load_history(
    csv_path: str | Path,
    date_col: str = "Date",
    target_col: str = DEFAULT_TARGET_NAME,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = sort_and_validate_timestamps(df, date_col=date_col)

    missing_cols = [c for c in TOTAL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required history columns: {missing_cols}")

    total_target = df[TOTAL_COLUMNS].sum(axis=1, min_count=len(TOTAL_COLUMNS))
    df[DEFAULT_TARGET_NAME] = total_target
    if target_col != DEFAULT_TARGET_NAME:
        df[target_col] = total_target

    df = ensure_airtemp_column(df)

    df.attrs["timeline_diagnostics"] = build_timeline_diagnostics(df[date_col])
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


def ensure_airtemp_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Airtemp" not in out.columns:
        if "AirTemp" in out.columns:
            out["Airtemp"] = out["AirTemp"]
        else:
            out["Airtemp"] = DEFAULT_AIRTEMP_VALUE
    elif "AirTemp" in out.columns:
        out["Airtemp"] = out["Airtemp"].fillna(out["AirTemp"])

    out["Airtemp"] = out["Airtemp"].fillna(DEFAULT_AIRTEMP_VALUE)
    return out


def _fill_weather_columns(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    weather_cols = [c for c in out.columns if c.startswith("Weather_")]
    if weather_cols:
        out = out.sort_values(date_col).reset_index(drop=True)
        out[weather_cols] = out[weather_cols].interpolate(method="linear", limit_direction="both")
        out[weather_cols] = out[weather_cols].ffill().bfill()
    return out


def attach_exogenous_columns(
    base_df: pd.DataFrame,
    weather: pd.DataFrame | None,
    date_col: str = "Date",
) -> pd.DataFrame:
    out = ensure_airtemp_column(base_df)
    if weather is not None:
        out = out.merge(weather, on=date_col, how="left")
        out = _fill_weather_columns(out, date_col=date_col)
    return out


def merge_weather_on_history(hist: pd.DataFrame, weather: pd.DataFrame | None, date_col: str = "Date") -> pd.DataFrame:
    return attach_exogenous_columns(hist, weather, date_col=date_col)


def slice_history_before_date(hist: pd.DataFrame, target_date: str, date_col: str = "Date") -> pd.DataFrame:
    target_start = pd.Timestamp(target_date)
    out = hist[hist[date_col] < target_start].copy()
    out = out.sort_values(date_col).reset_index(drop=True)
    return out


def extract_truth_for_day(hist: pd.DataFrame, target_date: str, date_col: str = "Date") -> pd.DataFrame:
    target_start = pd.Timestamp(target_date)
    target_end = target_start + pd.Timedelta(days=1)
    out = hist[(hist[date_col] >= target_start) & (hist[date_col] < target_end)].copy()
    out = out.sort_values(date_col).reset_index(drop=True)
    return out


def build_target_day_frame(
    target_date: str,
    weather: pd.DataFrame | None = None,
    date_col: str = "Date",
) -> pd.DataFrame:
    target_start = pd.Timestamp(target_date)
    dates = pd.date_range(target_start, periods=N_STEPS_PER_DAY, freq=FORECAST_FREQ)
    target_df = pd.DataFrame({date_col: dates})
    return attach_exogenous_columns(target_df, weather, date_col=date_col)


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
