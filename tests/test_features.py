import pandas as pd
import pytest

from smartgrid.features.engineering import build_feature_table, normalize_feature_config


def test_build_feature_table_adds_lags_and_calendar():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=144 * 8, freq="10min"),
            "Ptot_HA": 1.0,
            "Ptot_HEI_13RT": 2.0,
            "Ptot_HEI_5RNS": 3.0,
            "Ptot_RIZOMM": 4.0,
            "Airtemp": 15.0,
            "tot": 10.0,
        }
    )
    out, feature_cols = build_feature_table(df, holiday_dates=set(), special_dates=set())
    assert len(out) > 0
    assert "lag_d7" in feature_cols
    assert "minute_of_day" in feature_cols


def test_build_feature_table_adds_shifted_recent_dynamics_for_strict_mode():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=144 * 9, freq="10min"),
            "Ptot_HA": 1.0,
            "Ptot_HEI_13RT": 2.0,
            "Ptot_HEI_5RNS": 3.0,
            "Ptot_RIZOMM": 4.0,
            "Airtemp": 15.0,
            "tot": 10.0,
        }
    )

    out, feature_cols = build_feature_table(
        df,
        holiday_dates=set(),
        special_dates=set(),
        include_shifted_recent_dynamics=True,
        forecast_mode="strict_day_ahead",
    )

    assert len(out) > 0
    assert "prev_day_lag_t1" in feature_cols
    assert "prev_day_rolling_mean_6" in feature_cols


def test_normalize_feature_config_rejects_recent_dynamics_in_strict_mode():
    with pytest.raises(ValueError):
        normalize_feature_config(
            {
                "forecast_mode": "strict_day_ahead",
                "include_recent_dynamics": True,
            }
        )
