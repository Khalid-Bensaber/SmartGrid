from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from smartgrid.common.constants import FORECAST_FREQ, N_STEPS_PER_DAY


@dataclass(frozen=True, slots=True)
class GapInterval:
    start: pd.Timestamp
    end: pd.Timestamp
    missing_points: int


def _as_datetime_index(values: pd.Series | pd.Index | Iterable[pd.Timestamp]) -> pd.DatetimeIndex:
    if isinstance(values, pd.DatetimeIndex):
        return values
    if isinstance(values, pd.Series):
        return pd.DatetimeIndex(values)
    if isinstance(values, pd.Index):
        return pd.DatetimeIndex(values)
    return pd.DatetimeIndex(list(values))


def sort_and_validate_timestamps(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    duplicate_mask = out[date_col].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = out.loc[duplicate_mask, date_col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        sample = duplicates[:5]
        more = max(len(duplicates) - len(sample), 0)
        detail = ", ".join(sample)
        if more:
            detail = f"{detail}, ... (+{more} more)"
        raise ValueError(f"Duplicate timestamps detected in {date_col}: {detail}")

    return out


def build_complete_time_grid(
    timestamps: pd.Series | pd.Index | Iterable[pd.Timestamp],
    freq: str = FORECAST_FREQ,
) -> pd.DatetimeIndex:
    index = _as_datetime_index(timestamps).sort_values()
    if len(index) == 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    return pd.date_range(index.min(), index.max(), freq=freq)


def missing_timestamps(
    timestamps: pd.Series | pd.Index | Iterable[pd.Timestamp],
    freq: str = FORECAST_FREQ,
) -> pd.DatetimeIndex:
    index = _as_datetime_index(timestamps).sort_values()
    if len(index) == 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    return build_complete_time_grid(index, freq=freq).difference(index)


def detect_gap_intervals(
    timestamps: pd.Series | pd.Index | Iterable[pd.Timestamp],
    freq: str = FORECAST_FREQ,
) -> list[GapInterval]:
    index = _as_datetime_index(timestamps).sort_values()
    if len(index) < 2:
        return []

    step = pd.Timedelta(freq)
    gaps: list[GapInterval] = []
    previous = index[0]
    for current in index[1:]:
        diff = current - previous
        if diff > step:
            missing_points = int(diff // step) - 1
            gaps.append(
                GapInterval(
                    start=previous + step,
                    end=current - step,
                    missing_points=missing_points,
                )
            )
        previous = current
    return gaps


def assign_segment_ids(
    timestamps: pd.Series | pd.Index | Iterable[pd.Timestamp],
    freq: str = FORECAST_FREQ,
) -> pd.Series:
    index = _as_datetime_index(timestamps)
    if len(index) == 0:
        return pd.Series(dtype="int64")

    gap_starts = pd.Series(index).diff().gt(pd.Timedelta(freq)).fillna(False)
    return gap_starts.cumsum().astype(int)


def is_exact_timestamp_available(
    timestamps: pd.Series | pd.Index | Iterable[pd.Timestamp],
    ts: pd.Timestamp,
) -> bool:
    index = _as_datetime_index(timestamps)
    return pd.Timestamp(ts) in index


def lookup_exact_lag(
    series: pd.Series,
    ts: pd.Timestamp,
    delta: pd.Timedelta,
):
    return series.get(pd.Timestamp(ts) - delta)


def exact_window(
    series: pd.Series,
    *,
    end_ts: pd.Timestamp,
    periods: int,
    freq: str = FORECAST_FREQ,
) -> pd.Series:
    window_index = pd.date_range(end=pd.Timestamp(end_ts), periods=periods, freq=freq)
    return series.reindex(window_index)


def has_contiguous_window(
    timestamps: pd.Series | pd.Index | Iterable[pd.Timestamp],
    *,
    end_ts: pd.Timestamp,
    periods: int,
    freq: str = FORECAST_FREQ,
) -> bool:
    required = pd.date_range(end=pd.Timestamp(end_ts), periods=periods, freq=freq)
    return required.isin(_as_datetime_index(timestamps)).all()


def has_complete_day_coverage(
    timestamps: pd.Series | pd.Index | Iterable[pd.Timestamp],
    target_date: str | pd.Timestamp,
    *,
    periods: int = N_STEPS_PER_DAY,
    freq: str = FORECAST_FREQ,
) -> bool:
    day_start = pd.Timestamp(target_date).normalize()
    expected = pd.date_range(day_start, periods=periods, freq=freq)
    return expected.isin(_as_datetime_index(timestamps)).all()


def build_timeline_diagnostics(
    timestamps: pd.Series | pd.Index | Iterable[pd.Timestamp],
    freq: str = FORECAST_FREQ,
) -> dict[str, object]:
    index = _as_datetime_index(timestamps).sort_values()
    gaps = detect_gap_intervals(index, freq=freq)
    total_missing = int(sum(gap.missing_points for gap in gaps))
    largest_gap = max((gap.missing_points for gap in gaps), default=0)
    largest_gap_duration = (
        str(pd.Timedelta(freq) * largest_gap) if largest_gap > 0 else pd.Timedelta(0).__str__()
    )

    return {
        "row_count": int(len(index)),
        "date_min": str(index.min()) if len(index) else None,
        "date_max": str(index.max()) if len(index) else None,
        "duplicate_count": int(index.duplicated().sum()),
        "gap_count": int(len(gaps)),
        "missing_timestamp_count": total_missing,
        "segment_count": int(len(gaps) + (1 if len(index) else 0)),
        "largest_gap_missing_points": int(largest_gap),
        "largest_gap_duration": largest_gap_duration,
        "gap_intervals_preview": [
            {
                "start": str(gap.start),
                "end": str(gap.end),
                "missing_points": int(gap.missing_points),
            }
            for gap in gaps[:5]
        ],
    }
