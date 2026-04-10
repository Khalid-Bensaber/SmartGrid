from __future__ import annotations

import pandas as pd

from smartgrid.common.constants import DEFAULT_TARGET_NAME, N_STEPS_PER_DAY


def add_calendar_features(df: pd.DataFrame, holiday_dates: set, special_dates: set, date_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = out[date_col]

    out["day_of_year"] = dt.dt.dayofyear
    out["minute_of_day"] = dt.dt.hour * 60 + dt.dt.minute
    out["weekday"] = dt.dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["month"] = dt.dt.month
    out["date_only"] = dt.dt.date
    out["is_holiday"] = out["date_only"].isin(holiday_dates).astype(int)
    out["is_special_day"] = out["date_only"].isin(special_dates).astype(int)
    out.loc[out["is_special_day"] == 1, "is_holiday"] = 0
    return out


def add_manual_lag_features(df: pd.DataFrame, target_col: str, lag_days: list[int] | tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    for lag_day in lag_days:
        out[f"lag_d{lag_day}"] = out[target_col].shift(N_STEPS_PER_DAY * lag_day)
    return out


def build_feature_table(hist_df: pd.DataFrame, holiday_dates: set, special_dates: set, date_col: str = "Date", target_col: str = DEFAULT_TARGET_NAME, lag_days: list[int] | tuple[int, ...] = (7, 1, 2, 3, 4, 5, 6)):
    df = hist_df.copy()
    df = add_calendar_features(df, holiday_dates, special_dates, date_col)
    df = add_manual_lag_features(df, target_col, lag_days)

    feature_cols = [
        "day_of_year",
        "minute_of_day",
        "weekday",
        "is_weekend",
        "month",
        "is_holiday",
        "is_special_day",
        "Airtemp",
        *[f"lag_d{lag_day}" for lag_day in lag_days],
    ]
    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    return df, feature_cols
