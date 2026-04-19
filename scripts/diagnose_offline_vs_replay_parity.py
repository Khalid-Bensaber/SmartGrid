from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from smartgrid.common.constants import DEFAULT_TARGET_NAME
from smartgrid.common.utils import ensure_dir
from smartgrid.data.loaders import (
    load_history,
    load_holiday_sets,
    load_weather_history,
    merge_weather_on_history,
)
from smartgrid.evaluation.reporting import evaluate_forecast_frame
from smartgrid.features.engineering import (
    build_feature_table,
    build_forecast_feature_row,
    normalize_feature_config,
)
from smartgrid.inference.consumption import predict_from_feature_matrix
from smartgrid.inference.day_ahead import _build_target_feature_frame, build_forecast_runtime
from smartgrid.registry.model_registry import load_consumption_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose parity between offline held-out artifacts and runtime replay features."
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def resolve_bundle_dir(args: argparse.Namespace) -> Path:
    if args.bundle_dir is not None:
        return Path(args.bundle_dir)
    if args.run_id is None:
        raise ValueError("Pass either --run-id or --bundle-dir.")
    return Path(args.artifacts_root) / "runs" / "consumption" / args.run_id


def resolve_saved_backtest_path(summary: dict, bundle_dir: Path) -> Path | None:
    candidates = [
        summary.get("offline_test_backtest_csv"),
        summary.get("backtest_csv"),
        bundle_dir.parent.parent / "exports" / "consumption" / bundle_dir.name / "offline_test_backtest.csv",
        bundle_dir.parent.parent / "exports" / "consumption" / bundle_dir.name / "backtest.csv",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def rebuild_offline_day(summary: dict, target_date: str) -> tuple[pd.DataFrame, list[str]]:
    date_col = summary.get("date_col", "Date")
    target_col = summary.get("target_column", DEFAULT_TARGET_NAME)
    feature_config = normalize_feature_config(summary.get("feature_config") or {})

    hist = load_history(summary["historical_csv"], date_col=date_col, target_col=target_col)
    weather = load_weather_history(summary.get("weather_csv"), date_col=date_col)
    hist = merge_weather_on_history(hist, weather, date_col=date_col)

    if feature_config.get("include_calendar", True) or feature_config.get(
        "include_cyclical_time",
        False,
    ):
        holiday_dates, special_dates = load_holiday_sets(summary["holidays_xlsx"])
    else:
        holiday_dates, special_dates = set(), set()

    feature_df, feature_columns = build_feature_table(
        hist_df=hist,
        holiday_dates=holiday_dates,
        special_dates=special_dates,
        date_col=date_col,
        target_col=target_col,
        lag_days=feature_config.get("lag_days", [7, 1, 2, 3, 4, 5, 6]),
        include_calendar=feature_config.get("include_calendar", True),
        include_temperature=feature_config.get("include_temperature", True),
        include_manual_daily_lags=feature_config.get("include_manual_daily_lags", True),
        include_cyclical_time=feature_config.get("include_cyclical_time", False),
        include_lag_aggregates=feature_config.get("include_lag_aggregates", False),
        include_recent_dynamics=feature_config.get("include_recent_dynamics", False),
        include_shifted_recent_dynamics=feature_config.get("include_shifted_recent_dynamics", False),
        include_weather=feature_config.get("include_weather", False),
        weather_mode=feature_config.get("weather_mode"),
        weather_columns=feature_config.get("weather_columns"),
        forecast_mode=feature_config.get("forecast_mode"),
        keep_invalid=True,
        include_validity_columns=True,
    )
    day_start = pd.Timestamp(target_date)
    day_end = day_start + pd.Timedelta(days=1)
    offline_day = feature_df[
        (feature_df[date_col] >= day_start) & (feature_df[date_col] < day_end)
    ].copy()
    offline_day = offline_day.sort_values(date_col).reset_index(drop=True)
    return offline_day, feature_columns


def rebuild_runtime_day(summary: dict, bundle_dir: Path, target_date: str, device: str) -> tuple[pd.DataFrame, list[str]]:
    runtime = build_forecast_runtime(
        historical_csv=summary["historical_csv"],
        current_dir=bundle_dir,
        artifacts_root=bundle_dir.parent.parent.parent,
        weather_csv=summary.get("weather_csv"),
        holidays_xlsx=summary.get("holidays_xlsx"),
        benchmark_csv=summary.get("benchmark_csv"),
        device_request=device,
    )
    _, target_df, context_series, fallback_row = _build_target_feature_frame(runtime, target_date)

    runtime_rows: list[dict[str, object]] = []
    for _, target_row in target_df.iterrows():
        feature_row = build_forecast_feature_row(
            target_row=target_row,
            context_series=context_series,
            feature_columns=runtime.feature_columns,
            include_temperature=runtime.feature_config.get("include_temperature", True),
            include_manual_daily_lags=runtime.feature_config.get("include_manual_daily_lags", True),
            lag_days=runtime.feature_config.get("lag_days", [7, 1, 2, 3, 4, 5, 6]),
            include_lag_aggregates=runtime.feature_config.get("include_lag_aggregates", False),
            include_recent_dynamics=runtime.feature_config.get("include_recent_dynamics", False),
            include_shifted_recent_dynamics=runtime.feature_config.get(
                "include_shifted_recent_dynamics",
                False,
            ),
            include_weather=runtime.feature_config.get("include_weather", False),
            weather_mode=runtime.feature_config.get("weather_mode"),
            weather_columns=runtime.feature_config.get("weather_columns"),
            fallback_row=fallback_row,
        )
        feature_row[runtime.date_col] = target_row[runtime.date_col]
        runtime_rows.append(feature_row)

    runtime_day = pd.DataFrame(runtime_rows).sort_values(runtime.date_col).reset_index(drop=True)
    return runtime_day, runtime.feature_columns


def compare_feature_frames(
    offline_day: pd.DataFrame,
    runtime_day: pd.DataFrame,
    feature_columns: list[str],
    date_col: str,
) -> pd.DataFrame:
    if len(offline_day) != len(runtime_day):
        raise RuntimeError(
            "Offline and runtime feature frames do not have the same number of rows: "
            f"offline={len(offline_day)} runtime={len(runtime_day)}"
        )

    rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(offline_day[date_col].tolist()):
        for column in feature_columns:
            offline_value = offline_day.iloc[index][column]
            runtime_value = runtime_day.iloc[index][column]
            matches = bool(
                (pd.isna(offline_value) and pd.isna(runtime_value))
                or np.isclose(offline_value, runtime_value, equal_nan=True)
            )
            abs_delta = (
                None
                if pd.isna(offline_value) or pd.isna(runtime_value)
                else float(abs(float(offline_value) - float(runtime_value)))
            )
            rows.append(
                {
                    "Date": timestamp,
                    "feature": column,
                    "offline_value": offline_value,
                    "runtime_value": runtime_value,
                    "matches": matches,
                    "abs_delta": abs_delta,
                }
            )
    return pd.DataFrame(rows)


def build_prediction_comparison(
    bundle,
    offline_day: pd.DataFrame,
    runtime_day: pd.DataFrame,
    feature_columns: list[str],
    date_col: str,
    target_col: str,
    saved_backtest_day: pd.DataFrame | None,
    device: str,
) -> tuple[pd.DataFrame, dict[str, dict | None]]:
    offline_valid = offline_day.loc[offline_day["valid_for_training"]].copy().reset_index(drop=True)
    runtime_valid = runtime_day.copy().reset_index(drop=True)

    if len(offline_valid) != len(runtime_valid):
        raise RuntimeError(
            "Offline and runtime feature frames do not have the same number of valid target rows "
            f"for prediction: offline={len(offline_valid)} runtime={len(runtime_valid)}"
        )

    offline_predictions = predict_from_feature_matrix(
        bundle.model,
        bundle.x_scaler,
        bundle.y_scaler,
        offline_valid[feature_columns].to_numpy(dtype=float),
        device,
    )
    runtime_predictions = predict_from_feature_matrix(
        bundle.model,
        bundle.x_scaler,
        bundle.y_scaler,
        runtime_valid[feature_columns].to_numpy(dtype=float),
        device,
    )

    comparison = pd.DataFrame(
        {
            "Date": offline_valid[date_col].tolist(),
            "offline_prediction_current_code": offline_predictions,
            "runtime_prediction_current_code": runtime_predictions,
            "truth": offline_valid[target_col].tolist(),
        }
    )
    comparison["abs_delta_offline_vs_runtime"] = (
        comparison["offline_prediction_current_code"] - comparison["runtime_prediction_current_code"]
    ).abs()

    if saved_backtest_day is not None and not saved_backtest_day.empty:
        merged_saved = saved_backtest_day[[date_col, "Ptot_TOTAL_Forecast"]].rename(
            columns={
                date_col: "Date",
                "Ptot_TOTAL_Forecast": "saved_offline_backtest_prediction",
            }
        )
        comparison = comparison.merge(merged_saved, on="Date", how="left")
        comparison["abs_delta_offline_vs_saved_backtest"] = (
            comparison["offline_prediction_current_code"] - comparison["saved_offline_backtest_prediction"]
        ).abs()
        comparison["abs_delta_runtime_vs_saved_backtest"] = (
            comparison["runtime_prediction_current_code"] - comparison["saved_offline_backtest_prediction"]
        ).abs()

    metrics = {
        "offline_rebuilt_current_code": evaluate_forecast_frame(
            comparison.rename(
                columns={
                    "truth": "Ptot_TOTAL_Real",
                    "offline_prediction_current_code": "Ptot_TOTAL_Forecast",
                }
            )
        ),
        "runtime_rebuilt_current_code": evaluate_forecast_frame(
            comparison.rename(
                columns={
                    "truth": "Ptot_TOTAL_Real",
                    "runtime_prediction_current_code": "Ptot_TOTAL_Forecast",
                }
            )
        ),
        "saved_offline_backtest": None,
    }
    if "saved_offline_backtest_prediction" in comparison.columns:
        metrics["saved_offline_backtest"] = evaluate_forecast_frame(
            comparison.rename(
                columns={
                    "truth": "Ptot_TOTAL_Real",
                    "saved_offline_backtest_prediction": "Ptot_TOTAL_Forecast",
                }
            )
        )

    return comparison, metrics


def format_summary_text(summary: dict) -> str:
    feature_lines = [
        f"- {column}: max_abs_delta={details['max_abs_delta']} mismatched_rows={details['mismatched_rows']}"
        for column, details in summary["feature_mismatches"].items()
        if details["mismatched_rows"] > 0
    ]
    if not feature_lines:
        feature_lines = ["- none"]

    saved_block = "not available"
    if summary["prediction_deltas"]["saved_backtest"] is not None:
        saved_block = json.dumps(summary["prediction_deltas"]["saved_backtest"], indent=2)

    return "\n".join(
        [
            f"Run: {summary['run_id']}",
            f"Target date: {summary['target_date']}",
            f"Offline rows: {summary['offline_rows']}",
            f"Runtime rows: {summary['runtime_rows']}",
            f"Saved backtest rows: {summary['saved_backtest_rows']}",
            "",
            "Feature mismatches:",
            *feature_lines,
            "",
            "Prediction deltas:",
            json.dumps(summary["prediction_deltas"], indent=2),
            "",
            "Saved backtest deltas:",
            saved_block,
            "",
            "Metrics:",
            json.dumps(summary["metrics"], indent=2),
        ]
    )


def main() -> None:
    args = parse_args()
    bundle_dir = resolve_bundle_dir(args).resolve()
    bundle = load_consumption_bundle(bundle_dir, device=args.device)
    summary = bundle.summary or {}
    run_id = summary.get("run_id", bundle_dir.name)
    target_date = str(pd.Timestamp(args.target_date).date())
    target_col = summary.get("target_column", DEFAULT_TARGET_NAME)
    date_col = summary.get("date_col", "Date")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(args.artifacts_root) / "audits" / "parity" / f"{run_id}__{target_date}"
    )
    ensure_dir(output_dir)

    offline_day, offline_feature_columns = rebuild_offline_day(summary, target_date)
    runtime_day, runtime_feature_columns = rebuild_runtime_day(
        summary,
        bundle_dir,
        target_date,
        args.device,
    )
    if offline_feature_columns != runtime_feature_columns:
        raise RuntimeError(
            "Offline and runtime feature column lists differ:\n"
            f"offline={offline_feature_columns}\n"
            f"runtime={runtime_feature_columns}"
        )

    feature_comparison = compare_feature_frames(
        offline_day,
        runtime_day,
        offline_feature_columns,
        date_col,
    )

    saved_backtest_path = resolve_saved_backtest_path(summary, bundle_dir)
    saved_backtest_day = None
    if saved_backtest_path is not None:
        saved_backtest_day = pd.read_csv(saved_backtest_path)
        saved_backtest_day[date_col] = pd.to_datetime(saved_backtest_day[date_col], errors="coerce")
        saved_backtest_day = saved_backtest_day[
            (saved_backtest_day[date_col] >= pd.Timestamp(target_date))
            & (saved_backtest_day[date_col] < pd.Timestamp(target_date) + pd.Timedelta(days=1))
        ].copy()
        saved_backtest_day = saved_backtest_day.sort_values(date_col).reset_index(drop=True)

    prediction_comparison, metrics = build_prediction_comparison(
        bundle,
        offline_day,
        runtime_day,
        offline_feature_columns,
        date_col,
        target_col,
        saved_backtest_day,
        args.device,
    )

    feature_summary = {}
    for column, group in feature_comparison.groupby("feature", dropna=False):
        deltas = group["abs_delta"].dropna()
        feature_summary[column] = {
            "mismatched_rows": int((~group["matches"]).sum()),
            "max_abs_delta": float(deltas.max()) if not deltas.empty else 0.0,
        }

    prediction_deltas = {
        "offline_vs_runtime": {
            "mean_abs_delta": float(prediction_comparison["abs_delta_offline_vs_runtime"].mean()),
            "max_abs_delta": float(prediction_comparison["abs_delta_offline_vs_runtime"].max()),
        },
        "saved_backtest": None,
    }
    if "abs_delta_offline_vs_saved_backtest" in prediction_comparison.columns:
        prediction_deltas["saved_backtest"] = {
            "offline_vs_saved_mean_abs_delta": float(
                prediction_comparison["abs_delta_offline_vs_saved_backtest"].mean()
            ),
            "offline_vs_saved_max_abs_delta": float(
                prediction_comparison["abs_delta_offline_vs_saved_backtest"].max()
            ),
            "runtime_vs_saved_mean_abs_delta": float(
                prediction_comparison["abs_delta_runtime_vs_saved_backtest"].mean()
            ),
            "runtime_vs_saved_max_abs_delta": float(
                prediction_comparison["abs_delta_runtime_vs_saved_backtest"].max()
            ),
        }

    summary_payload = {
        "run_id": run_id,
        "target_date": target_date,
        "bundle_dir": str(bundle_dir),
        "offline_rows": int(len(offline_day)),
        "runtime_rows": int(len(runtime_day)),
        "saved_backtest_rows": int(len(saved_backtest_day)) if saved_backtest_day is not None else 0,
        "saved_backtest_path": str(saved_backtest_path) if saved_backtest_path is not None else None,
        "feature_columns": offline_feature_columns,
        "feature_mismatches": feature_summary,
        "prediction_deltas": prediction_deltas,
        "metrics": metrics,
        "official_evaluation_rule": "runtime_replay_forecast_is_official; offline_test_is_diagnostic",
    }

    feature_comparison_path = output_dir / "feature_comparison.csv"
    prediction_comparison_path = output_dir / "prediction_comparison.csv"
    summary_json_path = output_dir / "summary.json"
    summary_txt_path = output_dir / "summary.txt"

    feature_comparison.to_csv(feature_comparison_path, index=False)
    prediction_comparison.to_csv(prediction_comparison_path, index=False)
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    summary_txt_path.write_text(format_summary_text(summary_payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": run_id,
                "target_date": target_date,
                "feature_comparison_csv": str(feature_comparison_path.resolve()),
                "prediction_comparison_csv": str(prediction_comparison_path.resolve()),
                "summary_json": str(summary_json_path.resolve()),
                "summary_txt": str(summary_txt_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
