import pandas as pd

from smartgrid.features.engineering import build_feature_table


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
