import pandas as pd

from smartgrid.evaluation.metrics import build_metrics_df, compute_metrics_v2


def test_metrics_pipeline_shapes():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2025-01-01", periods=4, freq="10min"),
            "real": [100.0, 110.0, 120.0, 115.0],
            "pred": [102.0, 108.0, 119.0, 117.0],
        }
    ).set_index("Date")
    metrics_df = build_metrics_df(df, real_col="real", fc_col="pred")
    metrics = compute_metrics_v2(metrics_df)
    assert metrics["count"] == 4
    assert metrics["MAE"] >= 0
    assert "SMAPE%" in metrics
