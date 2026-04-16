from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from smartgrid.common.constants import (
    BASIC_WEATHER_COLUMNS,
    DEFAULT_TARGET_NAME,
    DEFAULT_WEATHER_COLUMNS,
    FORECAST_FREQ,
    INTRADAY_REFORECAST_MODE,
    IRRADIANCE_WEATHER_COLUMNS,
    STRICT_DAY_AHEAD_MODE,
)
from smartgrid.data.timeline import (
    assign_segment_ids,
    build_timeline_diagnostics,
    exact_window,
    lookup_exact_lag,
    sort_and_validate_timestamps,
)

RECENT_DYNAMICS_COLUMNS = [
    "lag_t1",
    "lag_t2",
    "lag_t3",
    "delta_t1",
    "delta_t2",
    "rolling_mean_6",
    "rolling_std_6",
]

SHIFTED_RECENT_DYNAMICS_COLUMNS = [
    "prev_day_lag_t1",
    "prev_day_lag_t2",
    "prev_day_lag_t3",
    "prev_day_delta_t1",
    "prev_day_delta_t2",
    "prev_day_rolling_mean_6",
    "prev_day_rolling_std_6",
]

VALIDITY_COLUMNS = [
    "segment_id",
    "valid_target",
    "valid_manual_lags",
    "valid_recent_window",
    "valid_shifted_recent_window",
    "valid_exogenous",
    "valid_for_training",
]


def add_calendar_features(
    df: pd.DataFrame,
    holiday_dates: set,
    special_dates: set,
    date_col: str,
) -> pd.DataFrame:
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
    out["minute_of_day_sin"] = out["minute_of_day"].apply(
        lambda x: math.sin(2 * math.pi * x / (24 * 60))
    )
    out["minute_of_day_cos"] = out["minute_of_day"].apply(
        lambda x: math.cos(2 * math.pi * x / (24 * 60))
    )
    out["day_of_year_sin"] = out["day_of_year"].apply(lambda x: math.sin(2 * math.pi * x / 366.0))
    out["day_of_year_cos"] = out["day_of_year"].apply(lambda x: math.cos(2 * math.pi * x / 366.0))
    return out


def _resolve_context_series(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    context_series: pd.Series | None = None,
) -> pd.Series:
    if context_series is not None:
        resolved = context_series.copy()
        if not isinstance(resolved.index, pd.DatetimeIndex):
            resolved.index = pd.DatetimeIndex(resolved.index)
        return resolved.sort_index()
    return pd.Series(df[target_col].to_numpy(), index=pd.DatetimeIndex(df[date_col]), name=target_col)


def _lookup_exact_values(context_series: pd.Series, lookup_timestamps: pd.DatetimeIndex):
    return context_series.reindex(pd.DatetimeIndex(lookup_timestamps)).to_numpy()


def _build_exact_window_frame(
    context_series: pd.Series,
    timestamps: pd.Series | pd.Index | pd.DatetimeIndex,
    *,
    periods: int,
    anchor_shift: pd.Timedelta = pd.Timedelta(0),
    prefix: str = "step",
) -> pd.DataFrame:
    ts_index = pd.DatetimeIndex(timestamps)
    step = pd.Timedelta(FORECAST_FREQ)
    data = {}
    for offset in range(1, periods + 1):
        lookup_index = ts_index - anchor_shift - (step * offset)
        data[f"{prefix}_{offset}"] = _lookup_exact_values(context_series, lookup_index)
    return pd.DataFrame(data, index=pd.RangeIndex(len(ts_index)))


def _valid_block(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(True, index=df.index, dtype=bool)
    return df[columns].notna().all(axis=1)


def add_manual_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lag_days: list[int] | tuple[int, ...],
    *,
    date_col: str = "Date",
    context_series: pd.Series | None = None,
) -> pd.DataFrame:
    out = df.copy()
    context = _resolve_context_series(
        out,
        date_col=date_col,
        target_col=target_col,
        context_series=context_series,
    )
    timestamps = pd.DatetimeIndex(out[date_col])

    for lag_day in lag_days:
        lookup_index = timestamps - pd.Timedelta(days=lag_day)
        out[f"lag_d{lag_day}"] = _lookup_exact_values(context, lookup_index)
    return out


def add_lag_aggregate_features(
    df: pd.DataFrame,
    lag_days: list[int] | tuple[int, ...],
) -> pd.DataFrame:
    out = df.copy()
    lag_cols = [f"lag_d{lag_day}" for lag_day in lag_days]
    if not lag_cols:
        return out

    complete_lags = out[lag_cols].notna().all(axis=1)
    out["lag_mean"] = out[lag_cols].mean(axis=1).where(complete_lags)
    out["lag_std"] = out[lag_cols].std(axis=1).where(complete_lags)
    out["lag_min"] = out[lag_cols].min(axis=1).where(complete_lags)
    out["lag_max"] = out[lag_cols].max(axis=1).where(complete_lags)
    return out


def add_recent_dynamics_features(
    df: pd.DataFrame,
    target_col: str,
    *,
    date_col: str = "Date",
    context_series: pd.Series | None = None,
) -> pd.DataFrame:
    out = df.copy()
    context = _resolve_context_series(
        out,
        date_col=date_col,
        target_col=target_col,
        context_series=context_series,
    )
    recent_window = _build_exact_window_frame(
        context,
        out[date_col],
        periods=6,
        prefix="recent",
    )
    complete_window = recent_window.notna().all(axis=1)

    out["lag_t1"] = recent_window["recent_1"]
    out["lag_t2"] = recent_window["recent_2"]
    out["lag_t3"] = recent_window["recent_3"]
    out["delta_t1"] = out["lag_t1"] - out["lag_t2"]
    out["delta_t2"] = out["lag_t2"] - out["lag_t3"]
    out["rolling_mean_6"] = recent_window.mean(axis=1).where(complete_window)
    out["rolling_std_6"] = recent_window.std(axis=1).where(complete_window)
    return out


def add_shifted_recent_dynamics_features(
    df: pd.DataFrame,
    target_col: str,
    *,
    date_col: str = "Date",
    context_series: pd.Series | None = None,
) -> pd.DataFrame:
    out = df.copy()
    context = _resolve_context_series(
        out,
        date_col=date_col,
        target_col=target_col,
        context_series=context_series,
    )
    shifted_window = _build_exact_window_frame(
        context,
        out[date_col],
        periods=6,
        anchor_shift=pd.Timedelta(days=1),
        prefix="shifted_recent",
    )
    complete_window = shifted_window.notna().all(axis=1)

    out["prev_day_lag_t1"] = shifted_window["shifted_recent_1"]
    out["prev_day_lag_t2"] = shifted_window["shifted_recent_2"]
    out["prev_day_lag_t3"] = shifted_window["shifted_recent_3"]
    out["prev_day_delta_t1"] = out["prev_day_lag_t1"] - out["prev_day_lag_t2"]
    out["prev_day_delta_t2"] = out["prev_day_lag_t2"] - out["prev_day_lag_t3"]
    out["prev_day_rolling_mean_6"] = shifted_window.mean(axis=1).where(complete_window)
    out["prev_day_rolling_std_6"] = shifted_window.std(axis=1).where(complete_window)
    return out


def resolve_forecast_mode(
    forecast_mode: str | None,
    *,
    include_recent_dynamics: bool = False,
) -> str:
    if forecast_mode is None:
        return INTRADAY_REFORECAST_MODE if include_recent_dynamics else STRICT_DAY_AHEAD_MODE

    valid_modes = {STRICT_DAY_AHEAD_MODE, INTRADAY_REFORECAST_MODE}
    if forecast_mode not in valid_modes:
        raise ValueError(
            f"Unknown forecast_mode={forecast_mode!r}. Expected one of {sorted(valid_modes)}."
        )
    return forecast_mode


def normalize_feature_config(feature_config: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = dict(feature_config or {})
    normalized.setdefault("include_calendar", True)
    normalized.setdefault("include_temperature", True)
    normalized.setdefault("include_manual_daily_lags", True)
    normalized.setdefault("include_cyclical_time", False)
    normalized.setdefault("include_lag_aggregates", False)
    normalized.setdefault("include_recent_dynamics", False)
    normalized.setdefault("include_shifted_recent_dynamics", False)
    normalized.setdefault("include_weather", False)
    normalized.setdefault("lag_days", [7, 1, 2, 3, 4, 5, 6])
    normalized["forecast_mode"] = resolve_forecast_mode(
        normalized.get("forecast_mode"),
        include_recent_dynamics=bool(normalized.get("include_recent_dynamics", False)),
    )

    if (
        normalized["forecast_mode"] == STRICT_DAY_AHEAD_MODE
        and normalized["include_recent_dynamics"]
    ):
        raise ValueError(
            "Strict day-ahead feature sets cannot use include_recent_dynamics because "
            "they depend on same-day target observations."
        )

    return normalized


def resolve_weather_columns(
    weather_mode: str | None,
    weather_columns: list[str] | None,
) -> list[str]:
    if weather_columns:
        return weather_columns
    if weather_mode == "basic":
        return BASIC_WEATHER_COLUMNS.copy()
    if weather_mode == "irradiance":
        return IRRADIANCE_WEATHER_COLUMNS.copy()
    if weather_mode == "all":
        return DEFAULT_WEATHER_COLUMNS.copy()
    return []


def _build_feature_diagnostics(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    timeline_diagnostics: dict[str, object],
) -> dict[str, object]:
    missing_feature_counts = {
        col: int(df[col].isna().sum()) for col in feature_cols if int(df[col].isna().sum()) > 0
    }
    return {
        "timeline": timeline_diagnostics,
        "samples": {
            "rows_before_filtering": int(len(df)),
            "rows_with_valid_target": int(df["valid_target"].sum()),
            "rows_failing_manual_lags": int((~df["valid_manual_lags"]).sum()),
            "rows_failing_recent_window": int((~df["valid_recent_window"]).sum()),
            "rows_failing_shifted_recent_window": int(
                (~df["valid_shifted_recent_window"]).sum()
            ),
            "rows_failing_exogenous": int((~df["valid_exogenous"]).sum()),
            "rows_kept": int(df["valid_for_training"].sum()),
            "rows_dropped": int((~df["valid_for_training"]).sum()),
        },
        "missing_feature_counts": missing_feature_counts,
    }


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
    include_shifted_recent_dynamics: bool = False,
    include_weather: bool = False,
    weather_mode: str | None = None,
    weather_columns: list[str] | None = None,
    forecast_mode: str | None = None,
    *,
    keep_invalid: bool = False,
    include_validity_columns: bool = False,
    return_diagnostics: bool = False,
):
    df = sort_and_validate_timestamps(hist_df, date_col=date_col)
    df["segment_id"] = assign_segment_ids(df[date_col], freq=FORECAST_FREQ).to_numpy()
    timeline_diagnostics = build_timeline_diagnostics(df[date_col], freq=FORECAST_FREQ)
    context_series = _resolve_context_series(df, date_col=date_col, target_col=target_col)
    feature_cols: list[str] = []
    resolved_forecast_mode = resolve_forecast_mode(
        forecast_mode,
        include_recent_dynamics=include_recent_dynamics,
    )

    if resolved_forecast_mode == STRICT_DAY_AHEAD_MODE and include_recent_dynamics:
        raise ValueError(
            "Strict day-ahead features cannot include recent intraday dynamics derived "
            "from target-day truth."
        )

    if include_calendar or include_cyclical_time:
        df = add_calendar_features(df, holiday_dates, special_dates, date_col)

    if include_calendar:
        feature_cols.extend(
            [
                "day_of_year",
                "minute_of_day",
                "weekday",
                "is_weekend",
                "month",
                "is_holiday",
                "is_special_day",
            ]
        )

    if include_cyclical_time:
        df = add_cyclical_time_features(df)
        feature_cols.extend(
            [
                "minute_of_day_sin",
                "minute_of_day_cos",
                "day_of_year_sin",
                "day_of_year_cos",
            ]
        )

    if include_temperature and "Airtemp" in df.columns:
        feature_cols.append("Airtemp")

    manual_feature_cols: list[str] = []
    if include_manual_daily_lags:
        df = add_manual_lag_features(
            df,
            target_col,
            lag_days,
            date_col=date_col,
            context_series=context_series,
        )
        manual_feature_cols.extend([f"lag_d{lag_day}" for lag_day in lag_days])
        feature_cols.extend(manual_feature_cols)

        if include_lag_aggregates:
            df = add_lag_aggregate_features(df, lag_days)
            manual_feature_cols.extend(["lag_mean", "lag_std", "lag_min", "lag_max"])
            feature_cols.extend(["lag_mean", "lag_std", "lag_min", "lag_max"])

    recent_feature_cols: list[str] = []
    if include_recent_dynamics:
        df = add_recent_dynamics_features(
            df,
            target_col,
            date_col=date_col,
            context_series=context_series,
        )
        recent_feature_cols = RECENT_DYNAMICS_COLUMNS.copy()
        feature_cols.extend(recent_feature_cols)

    shifted_recent_feature_cols: list[str] = []
    if include_shifted_recent_dynamics:
        df = add_shifted_recent_dynamics_features(
            df,
            target_col,
            date_col=date_col,
            context_series=context_series,
        )
        shifted_recent_feature_cols = SHIFTED_RECENT_DYNAMICS_COLUMNS.copy()
        feature_cols.extend(shifted_recent_feature_cols)

    if include_weather:
        selected_weather_cols = resolve_weather_columns(weather_mode, weather_columns)
        for col in selected_weather_cols:
            if col in df.columns:
                feature_cols.append(col)

    feature_cols = list(dict.fromkeys(feature_cols))
    exogenous_feature_cols = [
        col
        for col in feature_cols
        if col not in manual_feature_cols + recent_feature_cols + shifted_recent_feature_cols
    ]

    df["valid_target"] = df[target_col].notna()
    df["valid_manual_lags"] = _valid_block(df, manual_feature_cols)
    df["valid_recent_window"] = _valid_block(df, recent_feature_cols)
    df["valid_shifted_recent_window"] = _valid_block(df, shifted_recent_feature_cols)
    df["valid_exogenous"] = _valid_block(df, exogenous_feature_cols)
    df["valid_for_training"] = (
        df["valid_target"]
        & df["valid_manual_lags"]
        & df["valid_recent_window"]
        & df["valid_shifted_recent_window"]
        & df["valid_exogenous"]
    )

    diagnostics = _build_feature_diagnostics(
        df,
        feature_cols=feature_cols,
        timeline_diagnostics=timeline_diagnostics,
    )

    if not keep_invalid:
        df = df.loc[df["valid_for_training"]].copy()

    df = df.reset_index(drop=True)
    if not include_validity_columns:
        df = df.drop(columns=VALIDITY_COLUMNS, errors="ignore")

    df.attrs["feature_diagnostics"] = diagnostics
    df.attrs["timeline_diagnostics"] = timeline_diagnostics
    if return_diagnostics:
        return df, feature_cols, diagnostics
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


def build_temporal_feature_values(
    ts: pd.Timestamp,
    context_series: pd.Series,
    *,
    include_manual_daily_lags: bool = True,
    lag_days: list[int] | tuple[int, ...] = (7, 1, 2, 3, 4, 5, 6),
    include_lag_aggregates: bool = False,
    include_recent_dynamics: bool = False,
    include_shifted_recent_dynamics: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {}
    step = pd.Timedelta(FORECAST_FREQ)

    lag_values: dict[str, object] = {}
    if include_manual_daily_lags:
        for lag_day in lag_days:
            lag_key = f"lag_d{lag_day}"
            lag_value = lookup_exact_lag(context_series, ts, pd.Timedelta(days=lag_day))
            row[lag_key] = lag_value
            lag_values[lag_key] = lag_value

        if include_lag_aggregates and lag_values:
            lag_series = pd.Series(lag_values, dtype=float)
            complete_lags = lag_series.notna().all()
            row["lag_mean"] = float(lag_series.mean()) if complete_lags else pd.NA
            row["lag_std"] = float(lag_series.std()) if complete_lags else pd.NA
            row["lag_min"] = float(lag_series.min()) if complete_lags else pd.NA
            row["lag_max"] = float(lag_series.max()) if complete_lags else pd.NA

    if include_recent_dynamics:
        recent_window = exact_window(
            context_series,
            end_ts=pd.Timestamp(ts) - step,
            periods=6,
            freq=FORECAST_FREQ,
        )
        row["lag_t1"] = recent_window.iloc[-1]
        row["lag_t2"] = recent_window.iloc[-2]
        row["lag_t3"] = recent_window.iloc[-3]
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
        complete_recent_window = recent_window.notna().all()
        row["rolling_mean_6"] = float(recent_window.mean()) if complete_recent_window else pd.NA
        row["rolling_std_6"] = float(recent_window.std()) if complete_recent_window else pd.NA

    if include_shifted_recent_dynamics:
        previous_day_window = exact_window(
            context_series,
            end_ts=pd.Timestamp(ts) - pd.Timedelta(days=1) - step,
            periods=6,
            freq=FORECAST_FREQ,
        )
        row["prev_day_lag_t1"] = previous_day_window.iloc[-1]
        row["prev_day_lag_t2"] = previous_day_window.iloc[-2]
        row["prev_day_lag_t3"] = previous_day_window.iloc[-3]
        row["prev_day_delta_t1"] = (
            row["prev_day_lag_t1"] - row["prev_day_lag_t2"]
            if pd.notna(row["prev_day_lag_t1"]) and pd.notna(row["prev_day_lag_t2"])
            else pd.NA
        )
        row["prev_day_delta_t2"] = (
            row["prev_day_lag_t2"] - row["prev_day_lag_t3"]
            if pd.notna(row["prev_day_lag_t2"]) and pd.notna(row["prev_day_lag_t3"])
            else pd.NA
        )
        complete_previous_day_window = previous_day_window.notna().all()
        row["prev_day_rolling_mean_6"] = (
            float(previous_day_window.mean()) if complete_previous_day_window else pd.NA
        )
        row["prev_day_rolling_std_6"] = (
            float(previous_day_window.std()) if complete_previous_day_window else pd.NA
        )

    return row


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
    include_shifted_recent_dynamics: bool = False,
    include_weather: bool = False,
    weather_mode: str | None = None,
    weather_columns: list[str] | None = None,
    fallback_row: pd.Series | None = None,
) -> dict[str, float]:
    ts = pd.Timestamp(target_row["Date"])
    row = target_row.to_dict()

    if include_temperature and "Airtemp" not in row and fallback_row is not None:
        row["Airtemp"] = fallback_row.get("Airtemp")

    row.update(
        build_temporal_feature_values(
            ts,
            context_series,
            include_manual_daily_lags=include_manual_daily_lags,
            lag_days=lag_days,
            include_lag_aggregates=include_lag_aggregates,
            include_recent_dynamics=include_recent_dynamics,
            include_shifted_recent_dynamics=include_shifted_recent_dynamics,
        )
    )

    if include_weather:
        selected_weather_cols = resolve_weather_columns(weather_mode, weather_columns)
        for col in selected_weather_cols:
            if pd.isna(row.get(col)) and fallback_row is not None and col in fallback_row.index:
                row[col] = fallback_row[col]

    if include_temperature and pd.isna(row.get("Airtemp")) and fallback_row is not None:
        row["Airtemp"] = fallback_row.get("Airtemp")

    return {column: row.get(column, pd.NA) for column in feature_columns}
