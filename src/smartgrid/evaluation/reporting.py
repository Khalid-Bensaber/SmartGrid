from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from smartgrid.common.constants import (
    DEFAULT_TARGET_NAME,
    FORECAST_SCHEMA_COLUMNS,
    TOL_ABS,
    TOL_REL,
)
from smartgrid.evaluation.metrics import (
    build_metrics_df,
    compute_metrics_v2,
    compute_new_vs_old_comparison,
    seasonal_naive_weekly,
)


def pick_analysis_day(backtest: pd.DataFrame, benchmark: pd.DataFrame | None, date_col: str, requested_day: str | None):
    all_days = sorted(backtest[date_col].dt.date.unique())
    if benchmark is not None:
        bench_days = set(benchmark[date_col].dt.date.unique())
        intersection_days = [d for d in all_days if d in bench_days]
        if intersection_days:
            all_days = intersection_days
    if not all_days:
        raise RuntimeError("No valid day found in backtest / benchmark intersection.")

    if requested_day is not None:
        day = pd.to_datetime(requested_day).date()
        if day not in all_days:
            raise RuntimeError(f"Requested analysis day {day} is not available in valid test days.")
        return day
    return all_days[-1]


def make_notebook_export_legacy_schema(df_day: pd.DataFrame, date_col: str) -> pd.DataFrame:
    export = pd.DataFrame(
        {
            "name": "CONSO_Prevision_Data",
            "Date": df_day[date_col],
            "Ptot_HA_Forecast": np.nan,
            "Ptot_HEI_13RT_Forecast": np.nan,
            "Ptot_HEI_5RNS_Forecast": np.nan,
            "Ptot_Ilot_Forecast": df_day["Ptot_TOTAL_Forecast"].values,
            "Ptot_RIZOMM_Forecast": np.nan,
        }
    )
    return export[FORECAST_SCHEMA_COLUMNS]


def make_total_export(
    df_day: pd.DataFrame,
    date_col: str,
    target_col: str = DEFAULT_TARGET_NAME,
) -> pd.DataFrame:
    cols = [date_col, target_col, "Ptot_TOTAL_Forecast"]
    if "OldLegacy_TOTAL_Forecast" in df_day.columns:
        cols.append("OldLegacy_TOTAL_Forecast")
    out = df_day[cols].copy()
    out = out.rename(columns={target_col: "Ptot_TOTAL_Real"})
    return out


def save_json(path: str | Path, payload: dict) -> Path:
    output_path = Path(path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def build_backtest_outputs(
    test_df: pd.DataFrame,
    date_col: str,
    predictions: np.ndarray,
    benchmark: pd.DataFrame | None,
    target_col: str = DEFAULT_TARGET_NAME,
) -> pd.DataFrame:
    cols = [date_col, target_col]
    if "lag_d1" in test_df.columns:
        cols.append("lag_d1")
    backtest = test_df[cols].copy()
    backtest["Ptot_TOTAL_Forecast"] = predictions
    backtest["name"] = "CONSO_Prevision_Data"
    if benchmark is not None:
        backtest = backtest.merge(benchmark, on=date_col, how="left")
    return backtest


def evaluate_backtest(
    backtest: pd.DataFrame,
    date_col: str,
    target_col: str = DEFAULT_TARGET_NAME,
) -> dict:
    model_eval = build_metrics_df(
        merged=backtest[[date_col, target_col, "Ptot_TOTAL_Forecast"]].set_index(date_col),
        real_col=target_col,
        fc_col="Ptot_TOTAL_Forecast",
        tol_abs=TOL_ABS,
        tol_rel=TOL_REL,
    )

    naive_df = model_eval.copy()
    naive_df["Ptot_TOTAL_FC"] = seasonal_naive_weekly(model_eval["Ptot_TOTAL_REAL"], lag="7D")
    naive_eval = build_metrics_df(
        merged=naive_df.reset_index(),
        real_col="Ptot_TOTAL_REAL",
        fc_col="Ptot_TOTAL_FC",
        tol_abs=TOL_ABS,
        tol_rel=TOL_REL,
    ).set_index(date_col if date_col in naive_df.reset_index().columns else model_eval.index.name)

    idx_valid = naive_eval["Ptot_TOTAL_FC"].notna()
    naive_eval_aligned = naive_eval.loc[idx_valid]

    metrics_model = compute_metrics_v2(model_eval)
    metrics_naive_weekly = (
        compute_metrics_v2(naive_eval_aligned) if len(naive_eval_aligned) > 0 else None
    )
    metrics_old_legacy = None
    metrics_model_on_old_overlap = None
    comparison_new_vs_old = None
    old_overlap_count = 0

    if "OldLegacy_TOTAL_Forecast" in backtest.columns:
        overlap = backtest.dropna(subset=["OldLegacy_TOTAL_Forecast"]).copy()
        old_overlap_count = int(len(overlap))
        if len(overlap) > 0:
            old_eval = build_metrics_df(
                merged=overlap[[date_col, target_col, "OldLegacy_TOTAL_Forecast"]].set_index(date_col),
                real_col=target_col,
                fc_col="OldLegacy_TOTAL_Forecast",
                tol_abs=TOL_ABS,
                tol_rel=TOL_REL,
            )
            model_overlap_eval = build_metrics_df(
                merged=overlap[[date_col, target_col, "Ptot_TOTAL_Forecast"]].set_index(date_col),
                real_col=target_col,
                fc_col="Ptot_TOTAL_Forecast",
                tol_abs=TOL_ABS,
                tol_rel=TOL_REL,
            )
            metrics_old_legacy = compute_metrics_v2(old_eval)
            metrics_model_on_old_overlap = compute_metrics_v2(model_overlap_eval)
            comparison_new_vs_old = compute_new_vs_old_comparison(metrics_model_on_old_overlap, metrics_old_legacy)

    comparison = {
        "MASE (MAE_model / MAE_naive_weekly)": (
            metrics_model["MAE"] / metrics_naive_weekly["MAE"]
            if metrics_naive_weekly and metrics_naive_weekly["MAE"]
            else None
        ),
        "MAE_skill% vs naive_weekly": (
            100.0 * (1.0 - metrics_model["MAE"] / metrics_naive_weekly["MAE"])
            if metrics_naive_weekly and metrics_naive_weekly["MAE"]
            else None
        ),
        "RMSE_skill% vs naive_weekly": (
            100.0 * (1.0 - metrics_model["RMSE"] / metrics_naive_weekly["RMSE"])
            if metrics_naive_weekly and metrics_naive_weekly["RMSE"]
            else None
        ),
    }

    return {
        "metrics_model": metrics_model,
        "metrics_naive_weekly": metrics_naive_weekly,
        "metrics_old_legacy": metrics_old_legacy,
        "metrics_model_on_old_overlap": metrics_model_on_old_overlap,
        "comparison_new_vs_old": comparison_new_vs_old,
        "comparison": comparison,
        "old_overlap_count": old_overlap_count,
    }
