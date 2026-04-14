from __future__ import annotations

import math

import pandas as pd

from smartgrid.common.constants import (
    BASIC_WEATHER_COLUMNS,
    DEFAULT_TARGET_NAME,
    DEFAULT_WEATHER_COLUMNS,
    IRRADIANCE_WEATHER_COLUMNS,
    N_STEPS_PER_DAY,
)


def add_calendar_features(df: pd.DataFrame, holiday_dates: set, special_dates: set, date_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = out[date_col]

    out["day_of_year"] = dt.dt.dayofyear
    out["minute_of_day"] = dt.dt.hour * 60 + dt.dt.minute
    out["weekday"] = dt.dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["month"] = dt.dt.month
    out["date_only"] = out[date_col].dt.date
    out["is_holiday"] = out["date_only"].isin(holiday_dates).astype(int)
    out["is_special_day"] = out["date_only"].isin(special_dates).astype(int)
    out.loc[out["is_special_day"] == 1, "is_holiday"] = 0
    return out


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["minute_of_day_sin"] = out["minute_of_day"].apply(lambda x: math.sin(2 * math.pi * x / (24 * 60)))
    out["minute_of_day_cos"] = out["minute_of_day"].apply(lambda x: math.cos(2 * math.pi * x / (24 * 60)))
    out["day_of_year_sin"] = out["day_of_year"].apply(lambda x: math.sin(2 * math.pi * x / 366.0))
    out["day_of_year_cos"] = out["day_of_year"].apply(lambda x: math.cos(2 * math.pi * x / 366.0))
    return out


def add_manual_lag_features(df: pd.DataFrame, target_col: str, lag_days: list[int] | tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    for lag_day in lag_days:
        out[f"lag_d{lag_day}"] = out[target_col].shift(N_STEPS_PER_DAY * lag_day)
    return out


def add_lag_aggregate_features(df: pd.DataFrame, lag_days: list[int] | tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    lag_cols = [f"lag_d{lag_day}" for lag_day in lag_days]
    if not lag_cols:
        return out
    out["lag_mean"] = out[lag_cols].mean(axis=1)
    out["lag_std"] = out[lag_cols].std(axis=1)
    out["lag_min"] = out[lag_cols].min(axis=1)
    out["lag_max"] = out[lag_cols].max(axis=1)
    return out


def add_recent_dynamics_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out["lag_t1"] = out[target_col].shift(1)
    out["lag_t2"] = out[target_col].shift(2)
    out["lag_t3"] = out[target_col].shift(3)
    out["delta_t1"] = out[target_col].shift(1) - out[target_col].shift(2)
    out["delta_t2"] = out[target_col].shift(2) - out[target_col].shift(3)
    out["rolling_mean_6"] = out[target_col].shift(1).rolling(6).mean()
    out["rolling_std_6"] = out[target_col].shift(1).rolling(6).std()
    return out


def resolve_weather_columns(weather_mode: str | None, weather_columns: list[str] | None) -> list[str]:
    if weather_columns:
        return weather_columns
    if weather_mode == "basic":
        return BASIC_WEATHER_COLUMNS.copy()
    if weather_mode == "irradiance":
        return IRRADIANCE_WEATHER_COLUMNS.copy()
    if weather_mode == "all":
        return DEFAULT_WEATHER_COLUMNS.copy()
    return []


def build_feature_table(
    hist_df: pd.DataFrame,
    holiday_dates: set,
    special_dates: set,
    date_col: str = "Date",
    target_col: str = DEFAULT_TARGET_NAME,
    lag_days: list[int] | tuple[int, ...] = (7, 1, 2, 3, 4, 5, 6),
    include_calendar: bool = True,
    include_temperature: bool = True,
    include_manual_daily_lags: bool = True,
    include_cyclical_time: bool = False,
    include_lag_aggregates: bool = False,
    include_recent_dynamics: bool = False,
    include_weather: bool = False,
    weather_mode: str | None = None,
    weather_columns: list[str] | None = None,
):
    df = hist_df.copy()
    feature_cols: list[str] = []

    if include_calendar or include_cyclical_time:
        df = add_calendar_features(df, holiday_dates, special_dates, date_col)

    if include_calendar:
        feature_cols.extend([
            "day_of_year",
            "minute_of_day",
            "weekday",
            "is_weekend",
            "month",
            "is_holiday",
            "is_special_day",
        ])

    if include_cyclical_time:
        df = add_cyclical_time_features(df)
        feature_cols.extend([
            "minute_of_day_sin",
            "minute_of_day_cos",
            "day_of_year_sin",
            "day_of_year_cos",
        ])

    if include_temperature and "Airtemp" in df.columns:
        feature_cols.append("Airtemp")

    if include_manual_daily_lags:
        df = add_manual_lag_features(df, target_col, lag_days)
        feature_cols.extend([f"lag_d{lag_day}" for lag_day in lag_days])

        if include_lag_aggregates:
            df = add_lag_aggregate_features(df, lag_days)
            feature_cols.extend(["lag_mean", "lag_std", "lag_min", "lag_max"])

    if include_recent_dynamics:
        df = add_recent_dynamics_features(df, target_col)
        feature_cols.extend([
            "lag_t1",
            "lag_t2",
            "lag_t3",
            "delta_t1",
            "delta_t2",
            "rolling_mean_6",
            "rolling_std_6",
        ])

    if include_weather:
        selected_weather_cols = resolve_weather_columns(weather_mode, weather_columns)
        for col in selected_weather_cols:
            if col in df.columns:
                feature_cols.append(col)

    feature_cols = list(dict.fromkeys(feature_cols))
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    return df, feature_cols


def prepare_forecast_base_frame(
    target_df: pd.DataFrame,
    holiday_dates: set,
    special_dates: set,
    date_col: str = "Date",
    include_calendar: bool = True,
    include_cyclical_time: bool = False,
) -> pd.DataFrame:
    out = target_df.copy()
    if include_calendar or include_cyclical_time:
        out = add_calendar_features(out, holiday_dates, special_dates, date_col)
    if include_cyclical_time:
        out = add_cyclical_time_features(out)
    return out


def build_forecast_feature_row(
    target_row: pd.Series,
    context_series: pd.Series,
    feature_columns: list[str],
    *,
    include_temperature: bool = True,
    include_manual_daily_lags: bool = True,
    lag_days: list[int] | tuple[int, ...] = (7, 1, 2, 3, 4, 5, 6),
    include_lag_aggregates: bool = False,
    include_recent_dynamics: bool = False,
    include_weather: bool = False,
    weather_mode: str | None = None,
    weather_columns: list[str] | None = None,
    fallback_row: pd.Series | None = None,
) -> dict[str, float]:
    ts = pd.Timestamp(target_row["Date"])
    row = target_row.to_dict()

    if include_temperature and "Airtemp" not in row and fallback_row is not None:
        row["Airtemp"] = fallback_row.get("Airtemp")

    lag_values: dict[str, float] = {}
    if include_manual_daily_lags:
        for lag_day in lag_days:
            lag_value = context_series.get(ts - pd.Timedelta(days=lag_day))
            lag_key = f"lag_d{lag_day}"
            row[lag_key] = lag_value
            lag_values[lag_key] = lag_value

        if include_lag_aggregates and lag_values:
            lag_series = pd.Series(lag_values, dtype=float)
            row["lag_mean"] = float(lag_series.mean())
            row["lag_std"] = float(lag_series.std())
            row["lag_min"] = float(lag_series.min())
            row["lag_max"] = float(lag_series.max())

    if include_recent_dynamics:
        prev_values = context_series[context_series.index < ts].tail(6)
        row["lag_t1"] = prev_values.iloc[-1] if len(prev_values) >= 1 else pd.NA
        row["lag_t2"] = prev_values.iloc[-2] if len(prev_values) >= 2 else pd.NA
        row["lag_t3"] = prev_values.iloc[-3] if len(prev_values) >= 3 else pd.NA
        row["delta_t1"] = (
            row["lag_t1"] - row["lag_t2"]
            if pd.notna(row["lag_t1"]) and pd.notna(row["lag_t2"])
            else pd.NA
        )
        row["delta_t2"] = (
            row["lag_t2"] - row["lag_t3"]
            if pd.notna(row["lag_t2"]) and pd.notna(row["lag_t3"])
            else pd.NA
        )
        row["rolling_mean_6"] = float(prev_values.mean()) if len(prev_values) >= 1 else pd.NA
        row["rolling_std_6"] = float(prev_values.std()) if len(prev_values) >= 2 else pd.NA

    if include_weather:
        selected_weather_cols = resolve_weather_columns(weather_mode, weather_columns)
        for col in selected_weather_cols:
            if pd.isna(row.get(col)) and fallback_row is not None and col in fallback_row.index:
                row[col] = fallback_row[col]

    if include_temperature and pd.isna(row.get("Airtemp")) and fallback_row is not None:
        row["Airtemp"] = fallback_row.get("Airtemp")

    return {column: row.get(column, pd.NA) for column in feature_columns}
