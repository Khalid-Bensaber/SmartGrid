from __future__ import annotations

import pandas as pd


def chronological_split_by_ratio(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    n = len(df)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:i_train].copy()
    val_df = df.iloc[i_train:i_val].copy()
    test_df = df.iloc[i_val:].copy()
    return train_df, val_df, test_df


def _is_date_only_boundary(value: str) -> bool:
    return len(str(value).strip()) == 10


def _mask_at_or_before(df: pd.DataFrame, date_col: str, boundary: str) -> pd.Series:
    parsed = pd.to_datetime(boundary)
    if _is_date_only_boundary(boundary):
        return df[date_col] < parsed + pd.Timedelta(days=1)
    return df[date_col] <= parsed


def _mask_after(df: pd.DataFrame, date_col: str, boundary: str) -> pd.Series:
    parsed = pd.to_datetime(boundary)
    if _is_date_only_boundary(boundary):
        return df[date_col] >= parsed + pd.Timedelta(days=1)
    return df[date_col] > parsed


def chronological_split_by_dates(df: pd.DataFrame, date_col: str, train_end_date: str, val_end_date: str):
    train_mask = _mask_at_or_before(df, date_col, train_end_date)
    val_mask = _mask_after(df, date_col, train_end_date) & _mask_at_or_before(
        df,
        date_col,
        val_end_date,
    )
    test_mask = _mask_after(df, date_col, val_end_date)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Date-based split produced an empty split. Check train_end_date / val_end_date.")
    return train_df, val_df, test_df


def make_splits(df: pd.DataFrame, date_col: str, train_ratio: float, val_ratio: float, train_end_date: str | None = None, val_end_date: str | None = None):
    if train_end_date and val_end_date:
        return chronological_split_by_dates(df, date_col, train_end_date, val_end_date)
    return chronological_split_by_ratio(df, train_ratio=train_ratio, val_ratio=val_ratio)
