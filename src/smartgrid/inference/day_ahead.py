from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from smartgrid.common.constants import DEFAULT_TARGET_NAME
from smartgrid.common.paths import ForecastPaths, build_forecast_paths
from smartgrid.common.utils import get_device
from smartgrid.data.loaders import (
    build_target_day_frame,
    extract_truth_for_day,
    load_history,
    load_holiday_sets,
    load_weather_history,
    merge_weather_on_history,
    slice_history_before_date,
)
from smartgrid.features.engineering import (
    build_forecast_feature_row,
    prepare_forecast_base_frame,
    resolve_weather_columns,
)
from smartgrid.inference.consumption import predict_from_feature_dict
from smartgrid.registry.model_registry import ConsumptionBundle, load_current_consumption_bundle
from smartgrid.registry.model_registry import (
    list_ranked_consumption_bundle_dirs,
    load_consumption_bundle,
)


@dataclass(slots=True)
class ForecastRuntime:
    bundle: ConsumptionBundle
    device: torch.device
    historical_df: pd.DataFrame
    weather_df: pd.DataFrame | None
    holiday_dates: set
    special_dates: set
    feature_columns: list[str]
    feature_config: dict
    data_config: dict
    target_col: str
    date_col: str
    artifacts_root: Path
    current_dir: Path
    benchmark_csv: Path | None


def infer_target_date_from_history(hist_df: pd.DataFrame, date_col: str = "Date") -> str:
    last_timestamp = pd.Timestamp(hist_df[date_col].max())
    target_date = (last_timestamp.normalize() + pd.Timedelta(days=1)).date()
    return str(target_date)


def _resolve_data_config(bundle: ConsumptionBundle) -> dict:
    summary = bundle.summary or {}
    training_cfg = bundle.training_config or {}
    data_cfg = dict(training_cfg.get("data", {}))
    data_cfg.setdefault("date_col", summary.get("date_col", "Date"))
    data_cfg.setdefault("target_name", summary.get("target_column", DEFAULT_TARGET_NAME))
    if summary.get("weather_csv"):
        data_cfg["weather_csv"] = summary["weather_csv"]
    if summary.get("holidays_xlsx"):
        data_cfg["holidays_xlsx"] = summary["holidays_xlsx"]
    return data_cfg


def _resolve_feature_columns(bundle: ConsumptionBundle) -> list[str]:
    summary = bundle.summary or {}
    feature_columns = summary.get("feature_columns") or bundle.model_config.get("feature_columns")
    if not feature_columns:
        raise RuntimeError("Feature columns are missing from the promoted model bundle.")
    return list(feature_columns)


def _resolve_feature_config(bundle: ConsumptionBundle) -> dict:
    summary = bundle.summary or {}
    training_cfg = bundle.training_config or {}
    return dict(summary.get("feature_config") or training_cfg.get("features") or {})


def _fill_missing_target_exogenous(
    target_df: pd.DataFrame,
    feature_config: dict,
    fallback_row: pd.Series | None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    out = target_df.copy()
    required_cols: list[str] = []

    if feature_config.get("include_temperature", True):
        required_cols.append("Airtemp")

    if feature_config.get("include_weather", False):
        required_cols.extend(
            resolve_weather_columns(
                feature_config.get("weather_mode"),
                feature_config.get("weather_columns"),
            )
        )

    for col in required_cols:
        if col not in out.columns:
            out[col] = pd.NA
        fallback_value = None
        if fallback_row is not None and col in fallback_row.index and pd.notna(fallback_row[col]):
            fallback_value = fallback_row[col]
        if out[col].isna().any() and fallback_value is not None:
            out[col] = out[col].fillna(fallback_value)
            if logger is not None:
                logger.warning(
                    "Missing target-day exogenous values for %s. Falling back to last known value=%s.",
                    col,
                    fallback_value,
                )
    return out


def build_forecast_runtime(
    *,
    historical_csv: str | Path,
    current_dir: str | Path = "artifacts/models/consumption/current",
    artifacts_root: str | Path = "artifacts",
    weather_csv: str | Path | None = None,
    holidays_xlsx: str | Path | None = None,
    device_request: str = "auto",
    benchmark_csv: str | Path | None = "artifacts/benchmarks/consumption_feature_variants.csv",
    logger: logging.Logger | None = None,
) -> ForecastRuntime:
    device = get_device(device_request)
    bundle = load_current_consumption_bundle(current_dir, device)
    data_config = _resolve_data_config(bundle)
    feature_config = _resolve_feature_config(bundle)
    feature_columns = _resolve_feature_columns(bundle)
    date_col = data_config.get("date_col", "Date")
    target_col = data_config.get("target_name", DEFAULT_TARGET_NAME)

    hist_df = load_history(historical_csv, date_col=date_col, target_col=target_col)
    resolved_weather_csv = weather_csv if weather_csv is not None else data_config.get("weather_csv")
    weather_df = load_weather_history(resolved_weather_csv, date_col=date_col)
    hist_df = merge_weather_on_history(hist_df, weather_df, date_col=date_col)

    resolved_holidays_xlsx = holidays_xlsx if holidays_xlsx is not None else data_config.get("holidays_xlsx")
    if feature_config.get("include_calendar", True) or feature_config.get("include_cyclical_time", False):
        if resolved_holidays_xlsx is None:
            raise RuntimeError("Holidays workbook is required to rebuild the promoted feature set.")
        holiday_dates, special_dates = load_holiday_sets(resolved_holidays_xlsx)
    else:
        holiday_dates, special_dates = set(), set()

    if logger is not None:
        summary = bundle.summary or {}
        logger.info(
            "Loaded promoted model run_id=%s current_dir=%s device=%s n_features=%s",
            summary.get("run_id"),
            Path(current_dir),
            device,
            len(feature_columns),
        )
        logger.info(
            "Loaded history rows=%s date_range=[%s, %s]",
            len(hist_df),
            hist_df[date_col].min(),
            hist_df[date_col].max(),
        )

    return ForecastRuntime(
        bundle=bundle,
        device=device,
        historical_df=hist_df,
        weather_df=weather_df,
        holiday_dates=holiday_dates,
        special_dates=special_dates,
        feature_columns=feature_columns,
        feature_config=feature_config,
        data_config=data_config,
        target_col=target_col,
        date_col=date_col,
        artifacts_root=Path(artifacts_root),
        current_dir=Path(current_dir),
        benchmark_csv=Path(benchmark_csv) if benchmark_csv is not None else None,
    )


def _build_target_feature_frame(
    runtime: ForecastRuntime,
    target_date: str,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series | None]:
    target_date = str(pd.Timestamp(target_date).date())
    history_before = slice_history_before_date(runtime.historical_df, target_date, date_col=runtime.date_col)
    if history_before.empty:
        raise RuntimeError(f"No historical rows are available before target date {target_date}.")

    fallback_row = history_before.iloc[-1]
    target_df = build_target_day_frame(target_date, weather=runtime.weather_df, date_col=runtime.date_col)
    target_df = prepare_forecast_base_frame(
        target_df=target_df,
        holiday_dates=runtime.holiday_dates,
        special_dates=runtime.special_dates,
        date_col=runtime.date_col,
        include_calendar=runtime.feature_config.get("include_calendar", True),
        include_cyclical_time=runtime.feature_config.get("include_cyclical_time", False),
    )
    target_df = _fill_missing_target_exogenous(target_df, runtime.feature_config, fallback_row, logger=logger)
    context_series = history_before.set_index(runtime.date_col)[runtime.target_col].sort_index()
    return history_before, target_df, context_series, fallback_row


def collect_missing_features(runtime: ForecastRuntime, target_date: str) -> set[str]:
    _, target_df, context_series, fallback_row = _build_target_feature_frame(runtime, target_date)
    missing: set[str] = set()
    simulated_context = context_series.copy()

    for _, target_row in target_df.iterrows():
        feature_row = build_forecast_feature_row(
            target_row=target_row,
            context_series=simulated_context,
            feature_columns=runtime.feature_columns,
            include_temperature=runtime.feature_config.get("include_temperature", True),
            include_manual_daily_lags=runtime.feature_config.get("include_manual_daily_lags", True),
            lag_days=runtime.feature_config.get("lag_days", [7, 1, 2, 3, 4, 5, 6]),
            include_lag_aggregates=runtime.feature_config.get("include_lag_aggregates", False),
            include_recent_dynamics=runtime.feature_config.get("include_recent_dynamics", False),
            include_weather=runtime.feature_config.get("include_weather", False),
            weather_mode=runtime.feature_config.get("weather_mode"),
            weather_columns=runtime.feature_config.get("weather_columns"),
            fallback_row=fallback_row,
        )
        row_missing = [column for column, value in feature_row.items() if pd.isna(value)]
        missing.update(row_missing)

        # Simulate recursive day-ahead availability for recent-dynamics models.
        timestamp = pd.Timestamp(target_row[runtime.date_col])
        if timestamp not in simulated_context.index:
            simulated_context.loc[timestamp] = 0.0

    return missing


def select_runtime_for_target_date(
    runtime: ForecastRuntime,
    target_date: str,
    logger: logging.Logger | None = None,
) -> ForecastRuntime:
    current_missing = collect_missing_features(runtime, target_date)
    if not current_missing:
        return runtime

    if logger is not None:
        logger.warning(
            "Current promoted model run_id=%s is incompatible with target_date=%s missing_features=%s. Searching fallback model.",
            (runtime.bundle.summary or {}).get("run_id"),
            target_date,
            sorted(current_missing),
        )

    candidate_dirs = list_ranked_consumption_bundle_dirs(
        current_dir=runtime.current_dir,
        runs_dir=runtime.artifacts_root / "runs" / "consumption",
        benchmark_csv=runtime.benchmark_csv,
    )
    for bundle_dir in candidate_dirs:
        if bundle_dir.resolve() == runtime.bundle.bundle_dir.resolve():
            continue
        candidate_bundle = load_consumption_bundle(bundle_dir, runtime.device)
        candidate_runtime = ForecastRuntime(
            bundle=candidate_bundle,
            device=runtime.device,
            historical_df=runtime.historical_df,
            weather_df=runtime.weather_df,
            holiday_dates=runtime.holiday_dates,
            special_dates=runtime.special_dates,
            feature_columns=_resolve_feature_columns(candidate_bundle),
            feature_config=_resolve_feature_config(candidate_bundle),
            data_config=_resolve_data_config(candidate_bundle),
            target_col=_resolve_data_config(candidate_bundle).get("target_name", runtime.target_col),
            date_col=_resolve_data_config(candidate_bundle).get("date_col", runtime.date_col),
            artifacts_root=runtime.artifacts_root,
            current_dir=runtime.current_dir,
            benchmark_csv=runtime.benchmark_csv,
        )
        candidate_missing = collect_missing_features(candidate_runtime, target_date)
        if not candidate_missing:
            if logger is not None:
                logger.warning(
                    "Using fallback model run_id=%s bundle_dir=%s for target_date=%s",
                    (candidate_bundle.summary or {}).get("run_id"),
                    bundle_dir,
                    target_date,
                )
            return candidate_runtime

    raise RuntimeError(
        "No compatible registered model found for target date "
        f"{target_date}. Current missing features: {sorted(current_missing)}"
    )


def forecast_target_day(runtime: ForecastRuntime, target_date: str, logger: logging.Logger | None = None) -> pd.DataFrame:
    target_date = str(pd.Timestamp(target_date).date())
    runtime = select_runtime_for_target_date(runtime, target_date, logger=logger)
    _, target_df, context_series, fallback_row = _build_target_feature_frame(runtime, target_date, logger=logger)
    predictions: list[float] = []

    generated_at = datetime.now(timezone.utc).isoformat()
    model_run_id = (runtime.bundle.summary or {}).get("run_id", "unknown")
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
            include_weather=runtime.feature_config.get("include_weather", False),
            weather_mode=runtime.feature_config.get("weather_mode"),
            weather_columns=runtime.feature_config.get("weather_columns"),
            fallback_row=fallback_row,
        )
        missing = [column for column, value in feature_row.items() if pd.isna(value)]
        if missing:
            raise RuntimeError(
                f"Cannot forecast {target_row[runtime.date_col]} because features are missing: {missing}"
            )

        prediction = predict_from_feature_dict(
            runtime.bundle.model,
            runtime.bundle.x_scaler,
            runtime.bundle.y_scaler,
            feature_row,
            runtime.feature_columns,
            runtime.device,
        )
        timestamp = pd.Timestamp(target_row[runtime.date_col])
        context_series.loc[timestamp] = prediction
        predictions.append(prediction)

    forecast_df = pd.DataFrame(
        {
            "Date": target_df[runtime.date_col],
            "Ptot_TOTAL_Forecast": predictions,
            "model_run_id": model_run_id,
            "generated_at": generated_at,
            "target_date": target_date,
        }
    )

    truth_df = extract_truth_for_day(runtime.historical_df, target_date, date_col=runtime.date_col)
    if not truth_df.empty:
        truth_df = truth_df[[runtime.date_col, runtime.target_col]].rename(
            columns={runtime.date_col: "Date", runtime.target_col: "Ptot_TOTAL_Real"}
        )
        forecast_df = forecast_df.merge(truth_df, on="Date", how="left")

    if logger is not None:
        logger.info(
            "Forecasted target_date=%s points=%s model_run_id=%s",
            target_date,
            len(forecast_df),
            model_run_id,
        )

    return forecast_df


def write_forecast_outputs(
    forecast_df: pd.DataFrame,
    artifacts_root: str | Path,
    target_date: str,
    run_id: str,
) -> ForecastPaths:
    paths = build_forecast_paths(artifacts_root, target_date, run_id)
    forecast_df.to_csv(paths.current_output_path, index=False)
    forecast_df.to_csv(paths.archive_output_path, index=False)
    return paths
