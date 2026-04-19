from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from smartgrid.common.constants import (
    DEFAULT_TARGET_NAME,
    STRICT_DAY_AHEAD_MODE,
)
from smartgrid.common.profiling import build_runtime_diagnostics, maybe_cuda_synchronize
from smartgrid.common.paths import ForecastPaths, build_forecast_paths
from smartgrid.common.utils import get_device
from smartgrid.data.catalog import resolve_consumption_data_config
from smartgrid.data.loaders import (
    build_target_day_frame,
    extract_truth_for_day,
    load_history,
    load_holiday_sets,
    load_weather_history,
    merge_weather_on_history,
    slice_history_before_date,
)
from smartgrid.data.timeline import build_timeline_diagnostics, has_complete_day_coverage
from smartgrid.features.engineering import (
    build_forecast_feature_row,
    normalize_feature_config,
    prepare_forecast_base_frame,
    resolve_weather_columns,
)
from smartgrid.inference.consumption import predict_from_feature_dict
from smartgrid.registry.model_registry import (
    ConsumptionBundle,
    list_ranked_consumption_bundle_dirs,
    load_consumption_bundle,
    load_current_consumption_bundle,
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
    forecast_mode: str
    data_config: dict
    target_col: str
    date_col: str
    artifacts_root: Path
    current_dir: Path
    benchmark_csv: Path | None
    allow_fallback: bool


@dataclass(slots=True)
class ReplaySummary:
    replay_df: pd.DataFrame
    start_date: str
    end_date: str
    requested_model_run_id: str
    effective_model_run_ids: list[str]
    fallback_used: bool
    skipped_days: list[dict[str, str]]


@dataclass(slots=True)
class ForecastProfile:
    target_date: str
    timings_sec: dict[str, float]
    points: int
    model_run_id: str


@dataclass(slots=True)
class ReplayProfile:
    summary: ReplaySummary
    total_replay_sec: float
    per_day_sec: list[dict[str, float | str]]


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
    raw_feature_config = summary.get("feature_config") or training_cfg.get("features") or {}
    return normalize_feature_config(raw_feature_config)


def _resolve_runtime_data_config(
    bundle: ConsumptionBundle,
    *,
    dataset_key: str | None = None,
    catalog_path: str | Path | None = None,
    historical_csv: str | Path | None = None,
    weather_csv: str | Path | None = None,
    holidays_xlsx: str | Path | None = None,
) -> dict:
    bundle_data_config = _resolve_data_config(bundle)
    return resolve_consumption_data_config(
        bundle_data_config,
        dataset_key=dataset_key,
        catalog_path=catalog_path,
        overrides={
            "historical_csv": historical_csv,
            "weather_csv": weather_csv,
            "holidays_xlsx": holidays_xlsx,
        },
    )


def _require_strict_day_ahead_runtime(runtime: ForecastRuntime) -> None:
    if runtime.forecast_mode != STRICT_DAY_AHEAD_MODE:
        raise RuntimeError(
            "This flow only supports strict day-ahead forecasting. "
            f"Loaded bundle is forecast_mode={runtime.forecast_mode!r}."
        )


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
                    "Missing target-day exogenous values for %s. "
                    "Falling back to last known value=%s.",
                    col,
                    fallback_value,
                )
    return out


def build_forecast_runtime(
    *,
    historical_csv: str | Path | None = None,
    current_dir: str | Path = "artifacts/models/consumption/current",
    artifacts_root: str | Path = "artifacts",
    dataset_key: str | None = None,
    catalog_path: str | Path | None = None,
    weather_csv: str | Path | None = None,
    holidays_xlsx: str | Path | None = None,
    device_request: str = "auto",
    benchmark_csv: str | Path | None = None,
    allow_fallback: bool = False,
    logger: logging.Logger | None = None,
) -> ForecastRuntime:
    device = get_device(device_request)
    bundle = load_current_consumption_bundle(current_dir, device)
    feature_config = _resolve_feature_config(bundle)
    data_config = _resolve_runtime_data_config(
        bundle,
        dataset_key=dataset_key,
        catalog_path=catalog_path,
        historical_csv=historical_csv,
        weather_csv=weather_csv,
        holidays_xlsx=holidays_xlsx,
    )
    feature_columns = _resolve_feature_columns(bundle)
    date_col = data_config.get("date_col", "Date")
    target_col = data_config.get("target_name", DEFAULT_TARGET_NAME)
    forecast_mode = feature_config["forecast_mode"]

    hist_df = load_history(data_config["historical_csv"], date_col=date_col, target_col=target_col)
    weather_df = load_weather_history(data_config.get("weather_csv"), date_col=date_col)
    hist_df = merge_weather_on_history(hist_df, weather_df, date_col=date_col)

    if feature_config.get("include_calendar", True) or feature_config.get(
        "include_cyclical_time", False
    ):
        if data_config.get("holidays_xlsx") is None:
            raise RuntimeError("Holidays workbook is required to rebuild the promoted feature set.")
        holiday_dates, special_dates = load_holiday_sets(data_config["holidays_xlsx"])
    else:
        holiday_dates, special_dates = set(), set()

    if logger is not None:
        summary = bundle.summary or {}
        timeline_diagnostics = build_timeline_diagnostics(hist_df[date_col])
        runtime_diagnostics = build_runtime_diagnostics(
            requested_device=device_request,
            selected_device=device,
            profiling_enabled=False,
        )
        logger.info(
            "Loaded promoted model run_id=%s current_dir=%s device=%s n_features=%s",
            summary.get("run_id"),
            Path(current_dir),
            device,
            len(feature_columns),
        )
        logger.info(
            "Loaded dataset dataset_key=%s description=%s historical_csv=%s "
            "weather_csv=%s holidays_xlsx=%s forecast_mode=%s",
            data_config.get("dataset_key"),
            data_config.get("dataset_description"),
            data_config.get("historical_csv"),
            data_config.get("weather_csv"),
            data_config.get("holidays_xlsx"),
            forecast_mode,
        )
        logger.info(
            "Loaded history rows=%s date_range=[%s, %s]",
            len(hist_df),
            hist_df[date_col].min(),
            hist_df[date_col].max(),
        )
        logger.info(
            "History timeline gaps=%s missing_timestamps=%s segments=%s largest_gap=%s",
            timeline_diagnostics["gap_count"],
            timeline_diagnostics["missing_timestamp_count"],
            timeline_diagnostics["segment_count"],
            timeline_diagnostics["largest_gap_duration"],
        )
        logger.info("Inference runtime diagnostics=%s", runtime_diagnostics)

    return ForecastRuntime(
        bundle=bundle,
        device=device,
        historical_df=hist_df,
        weather_df=weather_df,
        holiday_dates=holiday_dates,
        special_dates=special_dates,
        feature_columns=feature_columns,
        feature_config=feature_config,
        forecast_mode=forecast_mode,
        data_config=data_config,
        target_col=target_col,
        date_col=date_col,
        artifacts_root=Path(artifacts_root),
        current_dir=Path(current_dir),
        benchmark_csv=Path(benchmark_csv) if benchmark_csv is not None else None,
        allow_fallback=allow_fallback,
    )


def _build_target_feature_frame(
    runtime: ForecastRuntime,
    target_date: str,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series | None]:
    target_date = str(pd.Timestamp(target_date).date())
    history_before = slice_history_before_date(
        runtime.historical_df,
        target_date,
        date_col=runtime.date_col,
    )
    if history_before.empty:
        raise RuntimeError(f"No historical rows are available before target date {target_date}.")

    fallback_row = history_before.iloc[-1]
    target_df = build_target_day_frame(
        target_date,
        weather=runtime.weather_df,
        date_col=runtime.date_col,
    )
    target_df = prepare_forecast_base_frame(
        target_df=target_df,
        holiday_dates=runtime.holiday_dates,
        special_dates=runtime.special_dates,
        date_col=runtime.date_col,
        include_calendar=runtime.feature_config.get("include_calendar", True),
        include_cyclical_time=runtime.feature_config.get("include_cyclical_time", False),
    )
    target_df = _fill_missing_target_exogenous(
        target_df,
        runtime.feature_config,
        fallback_row,
        logger=logger,
    )
    context_series = history_before.set_index(runtime.date_col)[runtime.target_col].sort_index()
    return history_before, target_df, context_series, fallback_row


def collect_missing_features(runtime: ForecastRuntime, target_date: str) -> set[str]:
    _require_strict_day_ahead_runtime(runtime)
    _, target_df, context_series, fallback_row = _build_target_feature_frame(runtime, target_date)
    missing: set[str] = set()

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
        row_missing = [column for column, value in feature_row.items() if pd.isna(value)]
        missing.update(row_missing)

    return missing


def select_runtime_for_target_date(
    runtime: ForecastRuntime,
    target_date: str,
    logger: logging.Logger | None = None,
) -> ForecastRuntime:
    _require_strict_day_ahead_runtime(runtime)
    current_missing = collect_missing_features(runtime, target_date)
    if not current_missing:
        return runtime

    if not runtime.allow_fallback:
        raise RuntimeError(
            "Current model is incompatible with target date "
            f"{target_date}. Missing features: {sorted(current_missing)}"
        )

    if logger is not None:
        logger.warning(
            "Current promoted model run_id=%s is incompatible with target_date=%s "
            "missing_features=%s. Searching fallback model.",
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
        candidate_feature_config = _resolve_feature_config(candidate_bundle)
        if candidate_feature_config["forecast_mode"] != runtime.forecast_mode:
            continue
        candidate_data_config = _resolve_runtime_data_config(candidate_bundle)
        candidate_runtime = ForecastRuntime(
            bundle=candidate_bundle,
            device=runtime.device,
            historical_df=runtime.historical_df,
            weather_df=runtime.weather_df,
            holiday_dates=runtime.holiday_dates,
            special_dates=runtime.special_dates,
            feature_columns=_resolve_feature_columns(candidate_bundle),
            feature_config=candidate_feature_config,
            forecast_mode=candidate_feature_config["forecast_mode"],
            data_config=candidate_data_config,
            target_col=candidate_data_config.get("target_name", runtime.target_col),
            date_col=candidate_data_config.get("date_col", runtime.date_col),
            artifacts_root=runtime.artifacts_root,
            current_dir=runtime.current_dir,
            benchmark_csv=runtime.benchmark_csv,
            allow_fallback=runtime.allow_fallback,
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


def forecast_target_day(
    runtime: ForecastRuntime,
    target_date: str,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    _require_strict_day_ahead_runtime(runtime)
    target_date = str(pd.Timestamp(target_date).date())
    runtime = select_runtime_for_target_date(runtime, target_date, logger=logger)
    _, target_df, context_series, fallback_row = _build_target_feature_frame(
        runtime,
        target_date,
        logger=logger,
    )
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
            include_shifted_recent_dynamics=runtime.feature_config.get(
                "include_shifted_recent_dynamics",
                False,
            ),
            include_weather=runtime.feature_config.get("include_weather", False),
            weather_mode=runtime.feature_config.get("weather_mode"),
            weather_columns=runtime.feature_config.get("weather_columns"),
            fallback_row=fallback_row,
        )
        missing = [column for column, value in feature_row.items() if pd.isna(value)]
        if missing:
            raise RuntimeError(
                f"Cannot forecast {target_row[runtime.date_col]} because features "
                f"are missing: {missing}"
            )

        prediction = predict_from_feature_dict(
            runtime.bundle.model,
            runtime.bundle.x_scaler,
            runtime.bundle.y_scaler,
            feature_row,
            runtime.feature_columns,
            runtime.device,
        )
        predictions.append(prediction)

    forecast_df = pd.DataFrame(
        {
            "Date": target_df[runtime.date_col],
            "Ptot_TOTAL_Forecast": predictions,
            "model_run_id": model_run_id,
            "generated_at": generated_at,
            "target_date": target_date,
            "dataset_key": runtime.data_config.get("dataset_key"),
            "forecast_mode": runtime.forecast_mode,
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


def forecast_next_day(
    runtime: ForecastRuntime,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    target_date = infer_target_date_from_history(runtime.historical_df, runtime.date_col)
    return forecast_target_day(runtime, target_date, logger=logger)


def profile_forecast_target_day(
    runtime: ForecastRuntime,
    target_date: str,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, ForecastProfile]:
    timings: dict[str, float] = {}
    target_date = str(pd.Timestamp(target_date).date())

    start = time.perf_counter()
    runtime = select_runtime_for_target_date(runtime, target_date, logger=logger)
    timings["runtime_selection_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    _, target_df, context_series, fallback_row = _build_target_feature_frame(
        runtime,
        target_date,
        logger=logger,
    )
    timings["target_day_feature_preparation_sec"] = time.perf_counter() - start

    predictions: list[float] = []
    generated_at = datetime.now(timezone.utc).isoformat()
    model_run_id = (runtime.bundle.summary or {}).get("run_id", "unknown")
    loop_start = time.perf_counter()
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
        missing = [column for column, value in feature_row.items() if pd.isna(value)]
        if missing:
            raise RuntimeError(
                f"Cannot forecast {target_row[runtime.date_col]} because features "
                f"are missing: {missing}"
            )
        pred_start = time.perf_counter()
        prediction = predict_from_feature_dict(
            runtime.bundle.model,
            runtime.bundle.x_scaler,
            runtime.bundle.y_scaler,
            feature_row,
            runtime.feature_columns,
            runtime.device,
        )
        maybe_cuda_synchronize(runtime.device)
        timings["model_inference_sec"] = timings.get("model_inference_sec", 0.0) + (
            time.perf_counter() - pred_start
        )
        predictions.append(prediction)
    timings["forecast_loop_sec"] = time.perf_counter() - loop_start

    forecast_df = pd.DataFrame(
        {
            "Date": target_df[runtime.date_col],
            "Ptot_TOTAL_Forecast": predictions,
            "model_run_id": model_run_id,
            "generated_at": generated_at,
            "target_date": target_date,
            "dataset_key": runtime.data_config.get("dataset_key"),
            "forecast_mode": runtime.forecast_mode,
        }
    )
    truth_df = extract_truth_for_day(runtime.historical_df, target_date, date_col=runtime.date_col)
    if not truth_df.empty:
        truth_df = truth_df[[runtime.date_col, runtime.target_col]].rename(
            columns={runtime.date_col: "Date", runtime.target_col: "Ptot_TOTAL_Real"}
        )
        forecast_df = forecast_df.merge(truth_df, on="Date", how="left")

    return forecast_df, ForecastProfile(
        target_date=target_date,
        timings_sec=timings,
        points=int(len(forecast_df)),
        model_run_id=model_run_id,
    )


def replay_forecast_period(
    runtime: ForecastRuntime,
    start_date: str,
    end_date: str,
    logger: logging.Logger | None = None,
) -> ReplaySummary:
    _require_strict_day_ahead_runtime(runtime)
    all_days = pd.date_range(start_date, end_date, freq="D")
    day_frames: list[pd.DataFrame] = []
    skipped_days: list[dict[str, str]] = []

    for day in all_days:
        target_date = str(day.date())
        truth_df = extract_truth_for_day(runtime.historical_df, target_date, date_col=runtime.date_col)
        if not has_complete_day_coverage(truth_df[runtime.date_col], target_date):
            reason = "Incomplete or missing target-day truth coverage."
            skipped_days.append({"target_date": target_date, "reason": reason})
            if logger is not None:
                logger.warning("Skipping replay target_date=%s: %s", target_date, reason)
            continue

        try:
            day_frames.append(forecast_target_day(runtime, target_date, logger=logger))
        except RuntimeError as exc:
            reason = str(exc)
            skipped_days.append({"target_date": target_date, "reason": reason})
            if logger is not None:
                logger.warning("Skipping replay target_date=%s: %s", target_date, reason)

    replay_df = pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame()
    requested_model_run_id = (runtime.bundle.summary or {}).get("run_id", "unknown")
    effective_model_run_ids = (
        sorted(str(x) for x in replay_df["model_run_id"].dropna().unique().tolist())
        if not replay_df.empty
        else []
    )
    fallback_used = any(model_id != requested_model_run_id for model_id in effective_model_run_ids)
    return ReplaySummary(
        replay_df=replay_df,
        start_date=str(pd.Timestamp(start_date).date()),
        end_date=str(pd.Timestamp(end_date).date()),
        requested_model_run_id=requested_model_run_id,
        effective_model_run_ids=effective_model_run_ids,
        fallback_used=fallback_used,
        skipped_days=skipped_days,
    )


def profile_replay_forecast_period(
    runtime: ForecastRuntime,
    start_date: str,
    end_date: str,
    logger: logging.Logger | None = None,
) -> ReplayProfile:
    all_days = pd.date_range(start_date, end_date, freq="D")
    per_day_sec: list[dict[str, float | str]] = []
    replay_start = time.perf_counter()
    day_frames: list[pd.DataFrame] = []
    skipped_days: list[dict[str, str]] = []

    for day in all_days:
        target_date = str(day.date())
        day_start = time.perf_counter()
        truth_df = extract_truth_for_day(runtime.historical_df, target_date, date_col=runtime.date_col)
        if not has_complete_day_coverage(truth_df[runtime.date_col], target_date):
            reason = "Incomplete or missing target-day truth coverage."
            skipped_days.append({"target_date": target_date, "reason": reason})
            per_day_sec.append({"target_date": target_date, "elapsed_sec": time.perf_counter() - day_start, "status": "skipped"})
            if logger is not None:
                logger.warning("Skipping replay target_date=%s: %s", target_date, reason)
            continue
        try:
            day_df, _ = profile_forecast_target_day(runtime, target_date, logger=logger)
            day_frames.append(day_df)
            per_day_sec.append({"target_date": target_date, "elapsed_sec": time.perf_counter() - day_start, "status": "ok"})
        except RuntimeError as exc:
            reason = str(exc)
            skipped_days.append({"target_date": target_date, "reason": reason})
            per_day_sec.append({"target_date": target_date, "elapsed_sec": time.perf_counter() - day_start, "status": "skipped"})
            if logger is not None:
                logger.warning("Skipping replay target_date=%s: %s", target_date, reason)

    replay_df = pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame()
    requested_model_run_id = (runtime.bundle.summary or {}).get("run_id", "unknown")
    effective_model_run_ids = (
        sorted(str(x) for x in replay_df["model_run_id"].dropna().unique().tolist())
        if not replay_df.empty
        else []
    )
    fallback_used = any(model_id != requested_model_run_id for model_id in effective_model_run_ids)
    summary = ReplaySummary(
        replay_df=replay_df,
        start_date=str(pd.Timestamp(start_date).date()),
        end_date=str(pd.Timestamp(end_date).date()),
        requested_model_run_id=requested_model_run_id,
        effective_model_run_ids=effective_model_run_ids,
        fallback_used=fallback_used,
        skipped_days=skipped_days,
    )
    return ReplayProfile(
        summary=summary,
        total_replay_sec=time.perf_counter() - replay_start,
        per_day_sec=per_day_sec,
    )


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
