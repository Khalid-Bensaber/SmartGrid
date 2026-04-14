import numpy as np
import pandas as pd

from smartgrid.evaluation.reporting import (
    build_backtest_outputs,
    evaluate_backtest,
    make_total_export,
)


def test_reporting_uses_configured_target_column():
    test_df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=4, freq="10min"),
            "custom_total": [100.0, 110.0, 120.0, 130.0],
        }
    )
    predictions = np.array([101.0, 109.0, 121.0, 129.0])

    backtest = build_backtest_outputs(
        test_df=test_df,
        date_col="Date",
        predictions=predictions,
        benchmark=None,
        target_col="custom_total",
    )
    total_export = make_total_export(backtest, "Date", target_col="custom_total")
    metrics = evaluate_backtest(backtest, "Date", target_col="custom_total")

    assert "Ptot_TOTAL_Real" in total_export.columns
    assert total_export["Ptot_TOTAL_Real"].tolist() == [100.0, 110.0, 120.0, 130.0]
    assert metrics["metrics_model"]["count"] == 4


def test_build_backtest_outputs_does_not_require_lag_d1():
    test_df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=2, freq="10min"),
            "tot": [10.0, 12.0],
        }
    )

    backtest = build_backtest_outputs(
        test_df=test_df,
        date_col="Date",
        predictions=np.array([11.0, 13.0]),
        benchmark=None,
    )

    assert "lag_d1" not in backtest.columns
    assert backtest["Ptot_TOTAL_Forecast"].tolist() == [11.0, 13.0]
