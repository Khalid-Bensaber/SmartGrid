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


def chronological_split_by_dates(df: pd.DataFrame, date_col: str, train_end_date: str, val_end_date: str):
    train_end = pd.to_datetime(train_end_date)
    val_end = pd.to_datetime(val_end_date)

    train_df = df[df[date_col] <= train_end].copy()
    val_df = df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    test_df = df[df[date_col] > val_end].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Date-based split produced an empty split. Check train_end_date / val_end_date.")
    return train_df, val_df, test_df


def make_splits(df: pd.DataFrame, date_col: str, train_ratio: float, val_ratio: float, train_end_date: str | None = None, val_end_date: str | None = None):
    if train_end_date and val_end_date:
        return chronological_split_by_dates(df, date_col, train_end_date, val_end_date)
    return chronological_split_by_ratio(df, train_ratio=train_ratio, val_ratio=val_ratio)
