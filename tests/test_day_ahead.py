import pandas as pd

from smartgrid.features.engineering import (
    build_forecast_feature_row,
    prepare_forecast_base_frame,
)
from smartgrid.inference.day_ahead import infer_target_date_from_history


def test_prepare_forecast_base_frame_adds_calendar_and_cyclical_columns():
    target_df = pd.DataFrame({"Date": pd.date_range("2025-01-02", periods=2, freq="10min")})

    out = prepare_forecast_base_frame(
        target_df=target_df,
        holiday_dates=set(),
        special_dates=set(),
        date_col="Date",
        include_calendar=True,
        include_cyclical_time=True,
    )

    assert "minute_of_day" in out.columns
    assert "weekday" in out.columns
    assert "minute_of_day_sin" in out.columns
    assert "day_of_year_cos" in out.columns


def test_build_forecast_feature_row_uses_only_past_context():
    context_index = pd.to_datetime(
        [
            "2025-01-01 00:10:00",
            "2025-01-01 23:40:00",
            "2025-01-01 23:50:00",
            "2025-01-02 00:00:00",
        ]
    )
    context_series = pd.Series([90.0, 95.0, 100.0, 110.0], index=context_index)
    target_row = pd.Series(
        {
            "Date": pd.Timestamp("2025-01-02 00:10:00"),
            "minute_of_day": 10,
            "Airtemp": 12.0,
        }
    )

    feature_row = build_forecast_feature_row(
        target_row=target_row,
        context_series=context_series,
        feature_columns=["minute_of_day", "Airtemp", "lag_d1", "lag_t1", "lag_t2", "delta_t1"],
        include_temperature=True,
        include_manual_daily_lags=True,
        lag_days=[1],
        include_recent_dynamics=True,
    )

    assert feature_row["lag_d1"] == 90.0
    assert feature_row["lag_t1"] == 110.0
    assert feature_row["lag_t2"] == 100.0
    assert feature_row["delta_t1"] == 10.0


def test_infer_target_date_from_history_uses_day_after_last_available_day():
    hist_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2025-01-05 00:10:00",
                    "2025-01-05 23:50:00",
                ]
            )
        }
    )

    assert infer_target_date_from_history(hist_df, "Date") == "2025-01-06"
