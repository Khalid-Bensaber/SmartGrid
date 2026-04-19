from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from smartgrid.data.loaders import build_target_day_frame, load_history, merge_weather_on_history
from smartgrid.data.splits import chronological_split_by_dates
from smartgrid.features.engineering import (
    build_feature_table,
    build_forecast_feature_row,
    prepare_forecast_base_frame,
)
from smartgrid.inference.day_ahead import ForecastRuntime, replay_forecast_period
from smartgrid.models.mlp import TorchMLP
from smartgrid.registry.model_registry import ConsumptionBundle


def _make_feature_history(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": dates,
            "tot": np.arange(len(dates), dtype=float),
            "Airtemp": 15.0,
        }
    )


def _make_replay_runtime(historical_df: pd.DataFrame) -> ForecastRuntime:
    model = TorchMLP(input_dim=1, hidden_layers=())
    with torch.no_grad():
        linear = model.net[0]
        linear.weight.zero_()
        linear.bias.fill_(0.5)

    x_scaler = MinMaxScaler().fit(np.array([[0.0], [1.0]], dtype=float))
    y_scaler = MinMaxScaler().fit(np.array([[0.0], [10.0]], dtype=float))
    bundle = ConsumptionBundle(
        bundle_dir=Path("."),
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        model_config={"feature_columns": ["lag_d1"]},
        summary={"run_id": "test_run"},
        training_config=None,
    )
    return ForecastRuntime(
        bundle=bundle,
        device=torch.device("cpu"),
        historical_df=historical_df,
        weather_df=None,
        holiday_dates=set(),
        special_dates=set(),
        feature_columns=["lag_d1"],
        feature_config={
            "forecast_mode": "strict_day_ahead",
            "include_temperature": False,
            "include_manual_daily_lags": True,
            "lag_days": [1],
            "include_lag_aggregates": False,
            "include_recent_dynamics": False,
            "include_shifted_recent_dynamics": False,
            "include_weather": False,
        },
        forecast_mode="strict_day_ahead",
        data_config={"dataset_key": "test"},
        target_col="tot",
        date_col="Date",
        artifacts_root=Path("artifacts"),
        current_dir=Path("artifacts/current"),
        benchmark_csv=None,
        allow_fallback=False,
    )


def test_build_feature_table_uses_exact_daily_lags_across_gaps():
    full_dates = pd.date_range("2025-01-01 00:00:00", periods=144 * 4, freq="10min")
    dates = full_dates[(full_dates < "2025-01-02") | (full_dates >= "2025-01-03")]
    hist = _make_feature_history(dates)

    shifted = hist.assign(row_shift_lag=hist["tot"].shift(144))
    target_ts = pd.Timestamp("2025-01-03 00:00:00")

    assert shifted.loc[shifted["Date"] == target_ts, "row_shift_lag"].iloc[0] == 0.0

    feature_df, _ = build_feature_table(
        hist,
        holiday_dates=set(),
        special_dates=set(),
        lag_days=[1],
        include_calendar=False,
        include_temperature=False,
        keep_invalid=True,
        include_validity_columns=True,
    )

    row = feature_df.loc[feature_df["Date"] == target_ts].iloc[0]
    assert pd.isna(row["lag_d1"])
    assert row["valid_target"]
    assert not row["valid_manual_lags"]
    assert not row["valid_for_training"]


def test_build_feature_table_uses_exact_timestamp_when_daily_lag_exists():
    dates = pd.date_range("2025-01-01 00:00:00", periods=144 * 3, freq="10min")
    hist = _make_feature_history(dates)

    feature_df, _ = build_feature_table(
        hist,
        holiday_dates=set(),
        special_dates=set(),
        lag_days=[1],
        include_calendar=False,
        include_temperature=False,
        keep_invalid=True,
        include_validity_columns=True,
    )

    target_ts = pd.Timestamp("2025-01-02 12:00:00")
    row = feature_df.loc[feature_df["Date"] == target_ts].iloc[0]
    expected = hist.loc[hist["Date"] == target_ts - pd.Timedelta(days=1), "tot"].iloc[0]

    assert row["lag_d1"] == expected
    assert row["valid_manual_lags"]


def test_build_feature_table_marks_recent_window_invalid_when_gap_crosses_window():
    dates = pd.date_range("2025-01-01 00:00:00", periods=10, freq="10min")
    dates = dates[dates != pd.Timestamp("2025-01-01 00:40:00")]
    hist = _make_feature_history(dates)

    feature_df, _ = build_feature_table(
        hist,
        holiday_dates=set(),
        special_dates=set(),
        include_calendar=False,
        include_temperature=False,
        include_manual_daily_lags=False,
        include_recent_dynamics=True,
        forecast_mode="intraday_reforecast",
        keep_invalid=True,
        include_validity_columns=True,
    )

    target_ts = pd.Timestamp("2025-01-01 01:30:00")
    row = feature_df.loc[feature_df["Date"] == target_ts].iloc[0]

    assert row["lag_t1"] == hist.loc[hist["Date"] == pd.Timestamp("2025-01-01 01:20:00"), "tot"].iloc[0]
    assert pd.isna(row["rolling_mean_6"])
    assert not row["valid_recent_window"]
    assert not row["valid_for_training"]


def test_training_and_inference_features_match_on_continuous_history():
    dates = pd.date_range("2025-01-01 00:00:00", periods=144 * 9, freq="10min")
    hist = _make_feature_history(dates)

    feature_df, feature_cols = build_feature_table(
        hist,
        holiday_dates=set(),
        special_dates=set(),
        lag_days=[1, 7],
        include_calendar=True,
        include_temperature=True,
        include_cyclical_time=True,
        include_shifted_recent_dynamics=True,
        forecast_mode="strict_day_ahead",
        keep_invalid=True,
        include_validity_columns=True,
    )

    target_ts = pd.Timestamp("2025-01-08 12:00:00")
    training_row = feature_df.loc[feature_df["Date"] == target_ts].iloc[0]
    assert training_row["valid_for_training"]

    target_base = prepare_forecast_base_frame(
        target_df=pd.DataFrame({"Date": [target_ts], "Airtemp": [training_row["Airtemp"]]}),
        holiday_dates=set(),
        special_dates=set(),
        date_col="Date",
        include_calendar=True,
        include_cyclical_time=True,
    )
    target_row = target_base.iloc[0]
    context_series = hist.loc[hist["Date"] < target_ts].set_index("Date")["tot"]

    forecast_row = build_forecast_feature_row(
        target_row=target_row,
        context_series=context_series,
        feature_columns=feature_cols,
        include_temperature=True,
        include_manual_daily_lags=True,
        lag_days=[1, 7],
        include_shifted_recent_dynamics=True,
    )

    for column in feature_cols:
        expected = training_row[column]
        actual = forecast_row[column]
        if pd.isna(expected):
            assert pd.isna(actual), column
        else:
            assert actual == pytest.approx(expected), column


def test_training_and_inference_features_match_with_weather_columns_and_default_airtemp():
    dates = pd.date_range("2025-01-01 00:00:00", periods=144 * 9, freq="10min")
    hist = pd.DataFrame(
        {
            "Date": dates,
            "tot": np.arange(len(dates), dtype=float),
            "Airtemp": 15.0,
        }
    )
    weather = pd.DataFrame(
        {
            "Date": dates,
            "Weather_AirTemp": np.linspace(5.0, 14.0, len(dates)),
            "Weather_CloudOpacity": np.linspace(0.0, 100.0, len(dates)),
        }
    )
    hist = merge_weather_on_history(hist, weather)

    feature_df, feature_cols = build_feature_table(
        hist,
        holiday_dates=set(),
        special_dates=set(),
        lag_days=[1, 7],
        include_calendar=False,
        include_temperature=True,
        include_weather=True,
        weather_mode="basic",
        keep_invalid=True,
        include_validity_columns=True,
    )

    target_ts = pd.Timestamp("2025-01-08 12:00:00")
    training_row = feature_df.loc[feature_df["Date"] == target_ts].iloc[0]
    assert training_row["valid_for_training"]
    assert training_row["Airtemp"] == pytest.approx(15.0)
    assert training_row["Weather_AirTemp"] != pytest.approx(training_row["Airtemp"])

    target_df = build_target_day_frame("2025-01-08", weather=weather)
    target_base = prepare_forecast_base_frame(
        target_df=target_df,
        holiday_dates=set(),
        special_dates=set(),
        date_col="Date",
        include_calendar=False,
        include_cyclical_time=False,
    )
    target_row = target_base.loc[target_base["Date"] == target_ts].iloc[0]
    context_series = hist.loc[hist["Date"] < target_ts].set_index("Date")["tot"]

    forecast_row = build_forecast_feature_row(
        target_row=target_row,
        context_series=context_series,
        feature_columns=feature_cols,
        include_temperature=True,
        include_manual_daily_lags=True,
        lag_days=[1, 7],
        include_weather=True,
        weather_mode="basic",
    )

    for column in feature_cols:
        expected = training_row[column]
        actual = forecast_row[column]
        if pd.isna(expected):
            assert pd.isna(actual), column
        else:
            assert actual == pytest.approx(expected), column


def test_replay_skips_days_with_incomplete_truth_coverage():
    full_dates = pd.date_range("2025-01-01 00:00:00", periods=144 * 5, freq="10min")
    dates = full_dates[(full_dates < "2025-01-04") | (full_dates >= "2025-01-05")]
    hist = _make_feature_history(dates)
    runtime = _make_replay_runtime(hist)

    replay = replay_forecast_period(runtime, "2025-01-03", "2025-01-04")

    assert sorted(replay.replay_df["target_date"].unique().tolist()) == ["2025-01-03"]
    assert len(replay.replay_df) == 144
    assert replay.skipped_days == [
        {
            "target_date": "2025-01-04",
            "reason": "Incomplete or missing target-day truth coverage.",
        }
    ]


def test_load_history_does_not_build_partial_targets(tmp_path):
    csv_path = tmp_path / "history.csv"
    pd.DataFrame(
        {
            "Date": ["2025-01-01 00:00:00", "2025-01-01 00:10:00"],
            "Ptot_HA": [1.0, 1.0],
            "Ptot_HEI_13RT": [2.0, None],
            "Ptot_HEI_5RNS": [3.0, 3.0],
            "Ptot_RIZOMM": [4.0, 4.0],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_history(csv_path)

    assert loaded.loc[0, "tot"] == pytest.approx(10.0)
    assert pd.isna(loaded.loc[1, "tot"])


def test_date_only_split_boundaries_include_the_full_day():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2025-01-01 23:50:00",
                    "2025-01-02 00:00:00",
                    "2025-01-02 23:50:00",
                    "2025-01-03 00:00:00",
                    "2025-01-03 23:50:00",
                    "2025-01-04 00:00:00",
                ]
            )
        }
    )

    train_df, val_df, test_df = chronological_split_by_dates(
        df,
        "Date",
        "2025-01-02",
        "2025-01-03",
    )

    assert train_df["Date"].tolist() == [
        pd.Timestamp("2025-01-01 23:50:00"),
        pd.Timestamp("2025-01-02 00:00:00"),
        pd.Timestamp("2025-01-02 23:50:00"),
    ]
    assert val_df["Date"].tolist() == [
        pd.Timestamp("2025-01-03 00:00:00"),
        pd.Timestamp("2025-01-03 23:50:00"),
    ]
    assert test_df["Date"].tolist() == [pd.Timestamp("2025-01-04 00:00:00")]


def test_timestamp_split_boundaries_remain_exact():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2025-01-02 00:00:00",
                    "2025-01-02 00:10:00",
                    "2025-01-03 00:00:00",
                    "2025-01-03 00:10:00",
                ]
            )
        }
    )

    train_df, val_df, test_df = chronological_split_by_dates(
        df,
        "Date",
        "2025-01-02 00:00:00",
        "2025-01-03 00:00:00",
    )

    assert train_df["Date"].tolist() == [pd.Timestamp("2025-01-02 00:00:00")]
    assert val_df["Date"].tolist() == [
        pd.Timestamp("2025-01-02 00:10:00"),
        pd.Timestamp("2025-01-03 00:00:00"),
    ]
    assert test_df["Date"].tolist() == [pd.Timestamp("2025-01-03 00:10:00")]
