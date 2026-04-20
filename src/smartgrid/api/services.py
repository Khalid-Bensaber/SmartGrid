from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from smartgrid.common.constants import DEFAULT_TARGET_NAME, STRICT_DAY_AHEAD_MODE
from smartgrid.common.logging import build_log_path, setup_logger
from smartgrid.common.paths import build_consumption_paths, build_replay_paths
from smartgrid.common.profiling import (
    TrainerProfiler,
    build_environment_summary,
    build_runtime_diagnostics,
)
from smartgrid.common.utils import ensure_dir, get_device, load_yaml, parse_hidden_layers, utc_run_id
from smartgrid.data.catalog import resolve_consumption_data_config
from smartgrid.data.loaders import (
    load_history,
    load_holiday_sets,
    load_old_benchmark,
    load_weather_history,
    merge_weather_on_history,
)
from smartgrid.data.splits import make_splits
from smartgrid.evaluation.metrics import compute_basic_metrics
from smartgrid.evaluation.reporting import (
    build_backtest_outputs,
    evaluate_backtest,
    evaluate_forecast_frame,
    make_notebook_export_legacy_schema,
    make_total_export,
    pick_analysis_day,
    save_json,
)
from smartgrid.features.engineering import build_feature_table, normalize_feature_config
from smartgrid.inference.consumption import predict_from_feature_dict
from smartgrid.inference.day_ahead import (
    build_forecast_runtime,
    forecast_next_day,
    forecast_target_day,
    replay_forecast_period,
    write_forecast_outputs,
)
from smartgrid.registry.model_registry import (
    ConsumptionBundle,
    list_ranked_consumption_bundle_dirs,
    load_consumption_bundle,
    load_current_consumption_bundle,
)
from smartgrid.training.artifacts import promote_bundle, save_training_bundle
from smartgrid.training.trainer import predict_model, train_mlp_regressor

REGISTRY_CURRENT_DIR = Path("artifacts/models/consumption/current")

_CURRENT_DIR_LOCKS: dict[Path, threading.Lock] = {}
_CURRENT_DIR_LOCKS_GUARD = threading.Lock()


class ApiServiceError(RuntimeError):
    def __init__(self, detail: Any, status_code: int = 400) -> None:
        super().__init__(str(detail))
        self.detail = detail
        self.status_code = status_code


def _to_builtin(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _rows_to_builtin(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_to_builtin(record) for record in frame.to_dict(orient="records")]


def _forecast_points(frame: pd.DataFrame) -> list[dict[str, Any]]:
    points = []
    for record in frame.to_dict(orient="records"):
        points.append(
            {
                "Date": str(record["Date"]),
                "Ptot_TOTAL_Forecast": float(record["Ptot_TOTAL_Forecast"]),
                "Ptot_TOTAL_Real": None
                if pd.isna(record.get("Ptot_TOTAL_Real"))
                else float(record["Ptot_TOTAL_Real"]),
            }
        )
    return points


def _resolve_current_dir(current_dir: str | None) -> Path:
    return Path(current_dir) if current_dir else REGISTRY_CURRENT_DIR


def _current_dir_lock_path(current_dir: Path) -> Path:
    return current_dir.resolve(strict=False)


@contextmanager
def guard_current_dir(current_dir: Path):
    lock_key = _current_dir_lock_path(current_dir)
    with _CURRENT_DIR_LOCKS_GUARD:
        lock = _CURRENT_DIR_LOCKS.setdefault(lock_key, threading.Lock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _service_logger(name: str, log_file: Path | None) -> Any:
    return setup_logger(name, log_file=log_file)


def _require_existing_bundle(current_dir: Path) -> None:
    if not current_dir.exists() or not (current_dir / "model.pt").exists():
        raise ApiServiceError("No promoted consumption model found", status_code=404)


def _build_runtime(
    *,
    dataset_key: str | None = None,
    catalog_path: str | None = None,
    historical_csv: str | None = None,
    weather_csv: str | None = None,
    holidays_xlsx: str | None = None,
    benchmark_csv: str | None = None,
    current_dir: str | None = None,
    artifacts_root: str = "artifacts",
    device: str = "auto",
    allow_fallback: bool = False,
    logger=None,
):
    resolved_current_dir = _resolve_current_dir(current_dir)
    _require_existing_bundle(resolved_current_dir)
    try:
        return build_forecast_runtime(
            current_dir=resolved_current_dir,
            artifacts_root=artifacts_root,
            dataset_key=dataset_key,
            catalog_path=catalog_path,
            historical_csv=historical_csv,
            weather_csv=weather_csv,
            holidays_xlsx=holidays_xlsx,
            benchmark_csv=benchmark_csv,
            device_request=device,
            allow_fallback=allow_fallback,
            logger=logger,
        )
    except FileNotFoundError as exc:
        raise ApiServiceError(str(exc), status_code=404) from exc
    except Exception as exc:
        raise ApiServiceError(str(exc), status_code=400) from exc


def _resolve_bundle_dir(model_ref: str, artifacts_root: Path) -> Path:
    raw = Path(model_ref)
    if raw.exists():
        return raw

    candidate = artifacts_root / "runs" / "consumption" / model_ref
    if candidate.exists():
        return candidate

    raise ApiServiceError(f"Unable to resolve model ref: {model_ref}", status_code=404)


def _resolve_bundle_summary(bundle_dir: Path) -> dict[str, Any]:
    for candidate in [bundle_dir / "run_summary.json", bundle_dir / "summary.json"]:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _resolve_bundle_forecast_mode(bundle_dir: Path) -> str | None:
    summary = _resolve_bundle_summary(bundle_dir)
    feature_cfg = summary.get("feature_config") or {}
    forecast_mode = summary.get("forecast_mode") or feature_cfg.get("forecast_mode")
    return str(forecast_mode) if forecast_mode else None


def _build_model_metadata(runtime, bundle_dir: Path) -> dict[str, Any]:
    summary = runtime.bundle.summary or {}
    config_path = summary.get("config_path")
    config_name = Path(config_path).name if config_path else None
    feature_cfg = summary.get("feature_config") or {}
    hidden_layers = summary.get("hidden_layers") or []
    return {
        "config_path": config_path,
        "config_name": config_name,
        "experiment_name": summary.get("experiment_name"),
        "dataset_key": summary.get("dataset_key"),
        "feature_config": feature_cfg,
        "forecast_mode": summary.get("forecast_mode") or feature_cfg.get("forecast_mode"),
        "feature_columns": summary.get("feature_columns"),
        "n_features": summary.get("n_features"),
        "hidden_layers": hidden_layers,
        "bundle_dir": str(bundle_dir.resolve()),
    }


def get_consumption_model_info(
    *,
    current_dir: str | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    resolved_current_dir = _resolve_current_dir(current_dir)
    with guard_current_dir(resolved_current_dir):
        _require_existing_bundle(resolved_current_dir)
        bundle = load_current_consumption_bundle(resolved_current_dir, get_device(device))
    summary = bundle.summary or {}
    return {
        "model_config": _to_builtin(bundle.model_config),
        "latest_summary": _to_builtin(summary),
        "dataset_key": summary.get("dataset_key"),
        "forecast_mode": (summary.get("feature_config") or {}).get("forecast_mode")
        or summary.get("forecast_mode"),
        "model_run_id": summary.get("run_id"),
        "current_dir": str(resolved_current_dir.resolve()),
    }


def list_consumption_models(
    *,
    current_dir: str | None = None,
    artifacts_root: str = "artifacts",
    benchmark_csv: str | None = None,
) -> dict[str, Any]:
    resolved_current_dir = _resolve_current_dir(current_dir)
    ranked_dirs = list_ranked_consumption_bundle_dirs(
        current_dir=resolved_current_dir,
        runs_dir=Path(artifacts_root) / "runs" / "consumption",
        benchmark_csv=benchmark_csv
        if benchmark_csv is not None
        else Path(artifacts_root) / "benchmarks" / "consumption_feature_variants.csv",
    )
    models = []
    for bundle_dir in ranked_dirs:
        summary = _resolve_bundle_summary(bundle_dir)
        models.append(
            {
                "bundle_dir": str(bundle_dir.resolve()),
                "run_id": summary.get("run_id", bundle_dir.name),
                "experiment_name": summary.get("experiment_name"),
                "dataset_key": summary.get("dataset_key"),
                "forecast_mode": summary.get("forecast_mode")
                or (summary.get("feature_config") or {}).get("forecast_mode"),
                "config_path": summary.get("config_path"),
                "is_current": bundle_dir.resolve(strict=False) == resolved_current_dir.resolve(strict=False),
            }
        )
    return {
        "current_dir": str(resolved_current_dir.resolve(strict=False)),
        "artifacts_root": str(Path(artifacts_root).resolve(strict=False)),
        "models": models,
    }


def predict_consumption_from_features_service(
    *,
    features: dict[str, float],
    current_dir: str | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    resolved_current_dir = _resolve_current_dir(current_dir)
    with guard_current_dir(resolved_current_dir):
        _require_existing_bundle(resolved_current_dir)
        bundle = load_current_consumption_bundle(resolved_current_dir, get_device(device))
        feature_columns = (
            bundle.summary["feature_columns"]
            if bundle.summary
            else bundle.model_config.get("feature_columns")
        )
        if feature_columns is None:
            raise ApiServiceError(
                "Feature columns missing from summary/model config",
                status_code=500,
            )

        missing = [column for column in feature_columns if column not in features]
        if missing:
            raise ApiServiceError({"missing_feature_columns": missing}, status_code=400)

        prediction = predict_from_feature_dict(
            bundle.model,
            bundle.x_scaler,
            bundle.y_scaler,
            features,
            feature_columns,
            get_device(device),
        )

    return {
        "prediction": float(prediction),
        "model_type": bundle.model_config.get("model_type", "unknown"),
        "feature_columns": list(feature_columns),
        "model_run_id": (bundle.summary or {}).get("run_id"),
        "current_dir": str(resolved_current_dir.resolve()),
    }


def run_consumption_forecast(
    *,
    target_date: str | None = None,
    dataset_key: str | None = None,
    catalog_path: str | None = None,
    historical_csv: str | None = None,
    weather_csv: str | None = None,
    holidays_xlsx: str | None = None,
    benchmark_csv: str | None = None,
    current_dir: str | None = None,
    artifacts_root: str = "artifacts",
    output_csv: str | None = None,
    device: str = "auto",
    allow_fallback: bool = False,
    write_outputs: bool = True,
) -> dict[str, Any]:
    resolved_current_dir = _resolve_current_dir(current_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    requested_target = target_date or "auto"
    logger = _service_logger(
        f"smartgrid.api.forecast.{stamp}",
        build_log_path(
            artifacts_root,
            "predict",
            f"{stamp}__forecast_{requested_target}.log",
        ),
    )

    with guard_current_dir(resolved_current_dir):
        runtime = _build_runtime(
            dataset_key=dataset_key,
            catalog_path=catalog_path,
            historical_csv=historical_csv,
            weather_csv=weather_csv,
            holidays_xlsx=holidays_xlsx,
            benchmark_csv=benchmark_csv,
            current_dir=str(resolved_current_dir),
            artifacts_root=artifacts_root,
            device=device,
            allow_fallback=allow_fallback,
            logger=logger,
        )
        requested_model_run_id = (runtime.bundle.summary or {}).get("run_id", "unknown")
        try:
            forecast_df = (
                forecast_target_day(runtime, target_date, logger=logger)
                if target_date
                else forecast_next_day(runtime, logger=logger)
            )
        except Exception as exc:
            raise ApiServiceError(str(exc), status_code=400) from exc

    effective_target_date = (
        str(forecast_df["target_date"].iloc[0]) if not forecast_df.empty else str(target_date or "")
    )
    effective_run_id = (
        str(forecast_df["model_run_id"].iloc[0]) if not forecast_df.empty else "unknown"
    )

    current_output_csv = None
    archive_output_csv = None
    if write_outputs:
        output_paths = write_forecast_outputs(
            forecast_df,
            artifacts_root,
            effective_target_date,
            effective_run_id,
        )
        current_output_csv = str(output_paths.current_output_path.resolve())
        archive_output_csv = str(output_paths.archive_output_path.resolve())

    custom_output_csv = None
    if output_csv is not None:
        custom_output = Path(output_csv)
        ensure_dir(custom_output.parent)
        forecast_df.to_csv(custom_output, index=False)
        custom_output_csv = str(custom_output.resolve())

    return {
        "target_date": effective_target_date,
        "points": _forecast_points(forecast_df),
        "model_run_id": effective_run_id,
        "requested_model_run_id": requested_model_run_id,
        "dataset_key": runtime.data_config.get("dataset_key"),
        "forecast_mode": runtime.forecast_mode,
        "fallback_used": effective_run_id != requested_model_run_id,
        "current_output_csv": current_output_csv,
        "archive_output_csv": archive_output_csv,
        "custom_output_csv": custom_output_csv,
    }


def run_consumption_replay(
    *,
    start_date: str,
    end_date: str,
    dataset_key: str | None = None,
    catalog_path: str | None = None,
    historical_csv: str | None = None,
    weather_csv: str | None = None,
    holidays_xlsx: str | None = None,
    benchmark_csv: str | None = None,
    current_dir: str | None = None,
    artifacts_root: str = "artifacts",
    device: str = "auto",
    allow_fallback: bool = False,
    write_per_day: bool = False,
    write_outputs: bool = True,
) -> dict[str, Any]:
    resolved_current_dir = _resolve_current_dir(current_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = _service_logger(
        f"smartgrid.api.replay.{stamp}",
        build_log_path(
            artifacts_root,
            "replay",
            f"{stamp}__{start_date}__{end_date}.log",
        ),
    )

    with guard_current_dir(resolved_current_dir):
        runtime = _build_runtime(
            dataset_key=dataset_key,
            catalog_path=catalog_path,
            historical_csv=historical_csv,
            weather_csv=weather_csv,
            holidays_xlsx=holidays_xlsx,
            benchmark_csv=benchmark_csv,
            current_dir=str(resolved_current_dir),
            artifacts_root=artifacts_root,
            device=device,
            allow_fallback=allow_fallback,
            logger=logger,
        )
        try:
            replay = replay_forecast_period(runtime, start_date, end_date, logger=logger)
        except Exception as exc:
            raise ApiServiceError(str(exc), status_code=400) from exc

    replay_df = replay.replay_df
    all_days = pd.date_range(replay.start_date, replay.end_date, freq="D")
    skipped_days = replay.skipped_days
    overall_metrics = evaluate_forecast_frame(replay_df)
    per_day_metrics = []
    per_day_model_usage = []
    if not replay_df.empty:
        for target_day, day_df in replay_df.groupby("target_date", dropna=False):
            metrics = evaluate_forecast_frame(day_df)
            if metrics is not None:
                per_day_metrics.append({"target_date": str(target_day), **metrics})
            per_day_model_usage.append(
                {
                    "target_date": str(target_day),
                    "model_run_ids": sorted(
                        str(x) for x in day_df["model_run_id"].dropna().unique().tolist()
                    ),
                }
            )

    output_csv_path = None
    metrics_json_path = None
    per_day_dir = None
    metrics_payload = {
        "start_date": replay.start_date,
        "end_date": replay.end_date,
        "n_days": int(len(all_days)),
        "n_rows": int(len(replay_df)),
        "requested_model_run_id": replay.requested_model_run_id,
        "effective_model_run_ids": replay.effective_model_run_ids,
        "fallback_enabled": bool(allow_fallback),
        "fallback_used": replay.fallback_used,
        "n_requested_days": int(len(all_days)),
        "n_forecasted_days": int(len(all_days) - len(skipped_days)),
        "n_skipped_days": int(len(skipped_days)),
        "skipped_days": skipped_days,
        "dataset_key": runtime.data_config.get("dataset_key"),
        "forecast_mode": runtime.forecast_mode,
        "overall_metrics": overall_metrics,
        "per_day_metrics": per_day_metrics,
        "per_day_model_usage": per_day_model_usage,
    }

    if write_outputs:
        replay_paths = build_replay_paths(artifacts_root, start_date, end_date, stamp)
        replay_df.to_csv(replay_paths.output_csv, index=False)
        output_csv_path = str(replay_paths.output_csv.resolve())

        if write_per_day:
            for target_day, day_forecast in replay_df.groupby("target_date", dropna=False):
                day_path = replay_paths.per_day_dir / f"forecast_{target_day}.csv"
                day_forecast.to_csv(day_path, index=False)
            per_day_dir = str(replay_paths.per_day_dir.resolve())

        metrics_payload["output_csv"] = output_csv_path
        save_json(replay_paths.metrics_json, metrics_payload)
        metrics_json_path = str(replay_paths.metrics_json.resolve())
        if per_day_dir is None:
            per_day_dir = str(replay_paths.per_day_dir.resolve())

    return {
        "start_date": replay.start_date,
        "end_date": replay.end_date,
        "points": _forecast_points(replay_df),
        "requested_model_run_id": replay.requested_model_run_id,
        "effective_model_run_ids": replay.effective_model_run_ids,
        "dataset_key": runtime.data_config.get("dataset_key"),
        "forecast_mode": runtime.forecast_mode,
        "overall_metrics": overall_metrics,
        "skipped_days": _to_builtin(skipped_days),
        "fallback_used": replay.fallback_used,
        "output_csv": output_csv_path,
        "metrics_json": metrics_json_path,
        "per_day_dir": per_day_dir,
        "n_days": int(len(all_days)),
        "n_rows": int(len(replay_df)),
        "n_requested_days": int(len(all_days)),
        "n_forecasted_days": int(len(all_days) - len(skipped_days)),
        "n_skipped_days": int(len(skipped_days)),
        "per_day_metrics": _to_builtin(per_day_metrics),
        "per_day_model_usage": _to_builtin(per_day_model_usage),
    }


def run_consumption_promote(
    *,
    run_id: str,
    artifacts_root: str = "artifacts",
) -> dict[str, Any]:
    logger = _service_logger(
        f"smartgrid.api.promote.{run_id}",
        build_log_path(artifacts_root, "train", f"promote__{run_id}.log"),
    )
    run_dir = Path(artifacts_root) / "runs" / "consumption" / run_id
    current_dir = Path(artifacts_root) / "models" / "consumption" / "current"
    if not run_dir.exists() or not (run_dir / "model.pt").exists():
        raise ApiServiceError(f"Run not found: {run_id}", status_code=404)

    with guard_current_dir(current_dir):
        promote_bundle(run_dir, current_dir)

    logger.info("Promoted %s -> %s", run_dir, current_dir)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "current_dir": str(current_dir.resolve()),
        "promoted": True,
    }


def run_consumption_training(
    *,
    config: str = "configs/consumption/mlp_strict_day_ahead_baseline.yaml",
    analysis_date: str | None = None,
    analysis_days: int = 1,
    resume_checkpoint: str | None = None,
    dataset_key: str | None = None,
    catalog_path: str | None = None,
    historical_csv: str | None = None,
    benchmark_csv: str | None = None,
    weather_csv: str | None = None,
    holidays_xlsx: str | None = None,
    promote: bool = False,
    profile: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    config_data = load_yaml(config)

    data_cfg = resolve_consumption_data_config(
        config_data["data"],
        dataset_key=dataset_key,
        catalog_path=catalog_path,
        overrides={
            "historical_csv": historical_csv,
            "benchmark_csv": benchmark_csv,
            "weather_csv": weather_csv,
            "holidays_xlsx": holidays_xlsx,
        },
    )
    split_cfg = config_data["split"]
    feat_cfg = normalize_feature_config(config_data["features"])
    train_cfg = config_data["training"]
    artifacts_cfg = config_data["artifacts"]
    target_col = data_cfg.get("target_name", DEFAULT_TARGET_NAME)

    run_id = utc_run_id("consumption_mlp")
    logger = _service_logger(
        f"smartgrid.api.train.{run_id}",
        build_log_path(artifacts_cfg["root_dir"], "train", f"{run_id}.log"),
    )
    paths = build_consumption_paths(
        root_dir=artifacts_cfg["root_dir"],
        exports_subdir=artifacts_cfg["exports_subdir"],
        registry_subdir=artifacts_cfg["registry_subdir"],
        run_id=run_id,
    )

    requested_device = device or train_cfg.get("device", "auto")
    selected_device = get_device(requested_device)
    hidden_layers = parse_hidden_layers(train_cfg["hidden_layers"])
    runtime_diagnostics = build_runtime_diagnostics(
        requested_device=requested_device,
        selected_device=selected_device,
        profiling_enabled=profile,
    )
    pipeline_timings: dict[str, float] = {}

    stage_start = time.perf_counter()
    holiday_dates, special_dates = load_holiday_sets(data_cfg["holidays_xlsx"])
    weather = load_weather_history(data_cfg.get("weather_csv"), date_col=data_cfg["date_col"])
    pipeline_timings["holiday_weather_load_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    hist = load_history(
        data_cfg["historical_csv"],
        date_col=data_cfg["date_col"],
        target_col=target_col,
    )
    hist = merge_weather_on_history(hist, weather, date_col=data_cfg["date_col"])
    pipeline_timings["data_loading_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    feat_df, feature_cols, feature_diagnostics = build_feature_table(
        hist_df=hist,
        holiday_dates=holiday_dates,
        special_dates=special_dates,
        date_col=data_cfg["date_col"],
        target_col=target_col,
        lag_days=feat_cfg.get("lag_days", [7, 1, 2, 3, 4, 5, 6]),
        include_calendar=feat_cfg.get("include_calendar", True),
        include_temperature=feat_cfg.get("include_temperature", True),
        include_manual_daily_lags=feat_cfg.get("include_manual_daily_lags", True),
        include_cyclical_time=feat_cfg.get("include_cyclical_time", False),
        include_lag_aggregates=feat_cfg.get("include_lag_aggregates", False),
        include_recent_dynamics=feat_cfg.get("include_recent_dynamics", False),
        include_shifted_recent_dynamics=feat_cfg.get("include_shifted_recent_dynamics", False),
        include_weather=feat_cfg.get("include_weather", False),
        weather_mode=feat_cfg.get("weather_mode"),
        weather_columns=feat_cfg.get("weather_columns"),
        forecast_mode=feat_cfg.get("forecast_mode"),
        return_diagnostics=True,
    )
    pipeline_timings["feature_engineering_sec"] = time.perf_counter() - stage_start
    timeline_summary = feature_diagnostics["timeline"]
    sample_summary = feature_diagnostics["samples"]

    stage_start = time.perf_counter()
    train_df, val_df, test_df = make_splits(
        feat_df,
        date_col=data_cfg["date_col"],
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        train_end_date=split_cfg.get("train_end_date"),
        val_end_date=split_cfg.get("val_end_date"),
    )
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[[target_col]].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df[[target_col]].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    X_val_scaled = x_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val)
    X_test_scaled = x_scaler.transform(X_test)
    pipeline_timings["split_scaling_prep_sec"] = time.perf_counter() - stage_start

    trainer_profiler = TrainerProfiler(enabled=profile)
    train_start = time.time()
    train_result = train_mlp_regressor(
        X_train=X_train_scaled,
        y_train=y_train_scaled,
        X_val=X_val_scaled,
        y_val=y_val_scaled,
        X_test=X_test_scaled,
        hidden_layers=hidden_layers,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        batch_size=train_cfg["batch_size"],
        epochs=train_cfg["epochs"],
        patience=train_cfg["patience"],
        dropout=train_cfg["dropout"],
        seed=train_cfg["seed"],
        num_workers=train_cfg["num_workers"],
        device=selected_device,
        resume_checkpoint=resume_checkpoint,
        logger=logger,
        profiler=trainer_profiler,
        batching_strategy=train_cfg.get("batching_strategy", "auto"),
        max_cuda_resident_bytes=int(train_cfg.get("max_cuda_resident_bytes", 512 * 1024 * 1024)),
    )
    train_duration_sec = time.time() - train_start
    pipeline_timings["dataloader_tensor_creation_sec"] = train_result.loader_prep_sec
    pipeline_timings["train_loop_total_sec"] = trainer_profiler.train_loop_total_sec
    pipeline_timings["validation_total_sec"] = trainer_profiler.validation_loop_total_sec

    stage_start = time.perf_counter()
    predictions = predict_model(train_result.model, train_result.test_x, y_scaler, selected_device)
    basic_metrics = compute_basic_metrics(y_test, predictions)

    benchmark = load_old_benchmark(data_cfg.get("benchmark_csv"), date_col=data_cfg["date_col"])
    backtest = build_backtest_outputs(
        test_df=test_df,
        date_col=data_cfg["date_col"],
        predictions=predictions,
        benchmark=benchmark,
        target_col=target_col,
    )
    evaluation = evaluate_backtest(
        backtest=backtest,
        date_col=data_cfg["date_col"],
        target_col=target_col,
    )
    pipeline_timings["post_train_evaluation_sec"] = time.perf_counter() - stage_start

    analysis_day = pick_analysis_day(backtest, benchmark, data_cfg["date_col"], analysis_date)
    start_day = np.datetime64(analysis_day)
    end_day = start_day + np.timedelta64(analysis_days, "D")
    mask = (backtest[data_cfg["date_col"]] >= start_day) & (
        backtest[data_cfg["date_col"]] < end_day
    )
    day_df = backtest.loc[mask].copy()

    notebook_export = make_notebook_export_legacy_schema(day_df, data_cfg["date_col"])
    notebook_export_path = paths.exports_dir / artifacts_cfg["notebook_output_filename"]
    notebook_export.to_csv(notebook_export_path, index=False)

    total_export = make_total_export(day_df, data_cfg["date_col"], target_col=target_col)
    total_export_path = paths.exports_dir / "total_forecast_consumption.csv"
    total_export.to_csv(total_export_path, index=False)
    offline_total_export_path = paths.exports_dir / "offline_test_total_forecast_consumption.csv"
    total_export.to_csv(offline_total_export_path, index=False)

    backtest_path = paths.exports_dir / "backtest.csv"
    offline_backtest_path = paths.exports_dir / "offline_test_backtest.csv"
    selected_day_path = paths.exports_dir / f"selected_day_{analysis_day}.csv"
    offline_selected_day_path = paths.exports_dir / f"offline_test_selected_day_{analysis_day}.csv"
    backtest.to_csv(backtest_path, index=False)
    backtest.to_csv(offline_backtest_path, index=False)
    day_df.to_csv(selected_day_path, index=False)
    day_df.to_csv(offline_selected_day_path, index=False)
    epochs_ran = len(train_result.history["train_loss"])
    best_val_loss = (
        float(min(train_result.history["val_loss"])) if train_result.history["val_loss"] else None
    )
    final_train_loss = (
        float(train_result.history["train_loss"][-1])
        if train_result.history["train_loss"]
        else None
    )
    final_val_loss = (
        float(train_result.history["val_loss"][-1]) if train_result.history["val_loss"] else None
    )

    summary = {
        "run_id": run_id,
        "problem": "consumption",
        "experiment_name": config_data.get("experiment_name"),
        "backend": "pytorch",
        "device": str(selected_device),
        "dataset_key": data_cfg.get("dataset_key"),
        "dataset_description": data_cfg.get("dataset_description"),
        "catalog_path": data_cfg.get("catalog_path"),
        "aliases": data_cfg.get("aliases") or {},
        "date_col": data_cfg["date_col"],
        "historical_csv": str(Path(data_cfg["historical_csv"]).resolve()),
        "holidays_xlsx": str(Path(data_cfg["holidays_xlsx"]).resolve()),
        "weather_csv": (
            str(Path(data_cfg["weather_csv"]).resolve()) if data_cfg.get("weather_csv") else None
        ),
        "benchmark_csv": (
            str(Path(data_cfg["benchmark_csv"]).resolve())
            if data_cfg.get("benchmark_csv")
            else None
        ),
        "target_column": target_col,
        "feature_columns": feature_cols,
        "feature_config": feat_cfg,
        "forecast_mode": feat_cfg.get("forecast_mode"),
        "timeline_diagnostics": timeline_summary,
        "feature_diagnostics": sample_summary,
        "feature_missingness": feature_diagnostics["missing_feature_counts"],
        "hidden_layers": list(hidden_layers),
        "n_features": int(len(feature_cols)),
        "batching_strategy": train_result.batching_strategy,
        "resident_data_bytes": int(train_result.resident_data_bytes),
        "train_duration_sec": train_duration_sec,
        "epochs_ran": epochs_ran,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "metrics_basic": basic_metrics,
        **evaluation,
        "offline_test_metrics": evaluation["metrics_model"],
        "offline_test_naive_weekly_metrics": evaluation["metrics_naive_weekly"],
        "offline_test_old_legacy_metrics": evaluation["metrics_old_legacy"],
        "offline_test_metrics_on_old_overlap": evaluation["metrics_model_on_old_overlap"],
        "offline_test_comparison": evaluation["comparison"],
        "n_history_rows": int(len(hist)),
        "n_rows_before_validity_filter": int(sample_summary["rows_before_filtering"]),
        "n_total_rows": int(len(feat_df)),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "train_date_min": str(train_df[data_cfg["date_col"]].min()),
        "train_date_max": str(train_df[data_cfg["date_col"]].max()),
        "val_date_min": str(val_df[data_cfg["date_col"]].min()),
        "val_date_max": str(val_df[data_cfg["date_col"]].max()),
        "test_date_min": str(test_df[data_cfg["date_col"]].min()),
        "test_date_max": str(test_df[data_cfg["date_col"]].max()),
        "selected_analysis_day": str(analysis_day),
        "analysis_days": analysis_days,
        "config_path": str(Path(config).resolve()),
        "history": train_result.history,
        "run_dir": str(paths.run_dir.resolve()),
        "exports_dir": str(paths.exports_dir.resolve()),
        "backtest_csv": str(backtest_path.resolve()),
        "offline_test_backtest_csv": str(offline_backtest_path.resolve()),
        "day_compare_csv": str(selected_day_path.resolve()),
        "offline_test_selected_day_csv": str(offline_selected_day_path.resolve()),
        "output_total_csv": str(total_export_path.resolve()),
        "offline_test_total_csv": str(offline_total_export_path.resolve()),
        "output_notebook_csv": str(notebook_export_path.resolve()),
        "evaluation_semantics": {
            "official_business_benchmark": "runtime_replay_forecast",
            "offline_training_evaluation": "diagnostic_offline_test_split",
            "legacy_backtest_csv_alias": "backtest.csv is a legacy alias for offline_test_backtest.csv",
        },
        "export_roles": {
            "backtest_csv": "legacy alias of offline_test_backtest_csv",
            "offline_test_backtest_csv": "diagnostic offline held-out evaluation output",
            "day_compare_csv": "legacy alias of offline_test_selected_day_csv",
            "offline_test_selected_day_csv": "selected analysis day extracted from the diagnostic offline test output",
            "output_total_csv": "legacy alias of offline_test_total_csv",
            "offline_test_total_csv": "compact selected-day export derived from the diagnostic offline test output",
            "output_notebook_csv": "legacy notebook compatibility export",
        },
    }

    stage_start = time.perf_counter()
    train_result.model_config["feature_columns"] = feature_cols
    save_training_bundle(
        model=train_result.model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        run_dir=paths.run_dir,
        model_config=train_result.model_config,
        run_summary=summary,
    )

    save_json(paths.exports_dir / artifacts_cfg["summary_filename"], summary)
    if promote:
        with guard_current_dir(paths.registry_current_dir):
            promote_bundle(paths.run_dir, paths.registry_current_dir)
    pipeline_timings["export_artifact_writing_sec"] = time.perf_counter() - stage_start
    summary["environment"] = build_environment_summary(selected_device, config, data_cfg)
    summary["runtime_diagnostics"] = runtime_diagnostics
    summary["profiling"] = {
        "pipeline_timings_sec": pipeline_timings,
        "trainer": trainer_profiler.to_summary(train_result.history),
    }
    summary["training_runtime"] = {
        "batching_strategy": train_result.batching_strategy,
        "resident_data_bytes": int(train_result.resident_data_bytes),
        "configured_batching_strategy": train_cfg.get("batching_strategy", "auto"),
        "max_cuda_resident_bytes": int(train_cfg.get("max_cuda_resident_bytes", 512 * 1024 * 1024)),
    }
    save_json(paths.run_dir / "run_summary.json", summary)
    save_json(paths.exports_dir / artifacts_cfg["summary_filename"], summary)

    payload = {
        "run_id": run_id,
        "run_dir": str(paths.run_dir.resolve()),
        "exports_dir": str(paths.exports_dir.resolve()),
        "promoted": bool(promote),
        "config": str(Path(config)),
        "experiment_name": config_data.get("experiment_name"),
        "selected_analysis_day": str(analysis_day),
        "train_duration_sec": train_duration_sec,
        "n_features": int(len(feature_cols)),
        "epochs_ran": epochs_ran,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "feature_config": feat_cfg,
        "forecast_mode": feat_cfg.get("forecast_mode"),
        "dataset_key": data_cfg.get("dataset_key"),
        "device": str(selected_device),
        "runtime_diagnostics": runtime_diagnostics,
        "batching_strategy": train_result.batching_strategy,
        "resident_data_bytes": int(train_result.resident_data_bytes),
        "test_date_min": str(test_df[data_cfg["date_col"]].min()),
        "test_date_max": str(test_df[data_cfg["date_col"]].max()),
        "offline_test_metrics": evaluation["metrics_model"],
        "metrics_model": evaluation["metrics_model"],
        "profiling_enabled": bool(profile),
    }
    return _to_builtin(payload)


def run_consumption_replay_benchmark(
    *,
    model_refs: list[str],
    start_date: str,
    end_date: str,
    dataset_key: str | None = None,
    catalog_path: str | None = None,
    historical_csv: str | None = None,
    weather_csv: str | None = None,
    holidays_xlsx: str | None = None,
    benchmark_csv: str | None = None,
    artifacts_root: str = "artifacts",
    device: str = "auto",
    allow_fallback: bool = False,
) -> dict[str, Any]:
    artifacts_root_path = Path(artifacts_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = _service_logger(
        f"smartgrid.api.replay_benchmark.{stamp}",
        build_log_path(
            artifacts_root_path,
            "replay",
            f"{stamp}__benchmark_replay__{start_date}__{end_date}.log",
        ),
    )

    out_dir = ensure_dir(
        artifacts_root_path / "benchmarks" / "replay" / f"{stamp}__{start_date}__{end_date}"
    )
    all_days = pd.date_range(start_date, end_date, freq="D")
    summary_rows = []
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "allow_fallback": bool(allow_fallback),
        "models": [],
        "skipped_models": [],
    }

    for model_ref in model_refs:
        bundle_dir = _resolve_bundle_dir(model_ref, artifacts_root_path)
        requested_run_id = bundle_dir.name
        forecast_mode = _resolve_bundle_forecast_mode(bundle_dir)
        if forecast_mode not in [None, STRICT_DAY_AHEAD_MODE]:
            manifest["skipped_models"].append(
                {
                    "requested_model_run_id": requested_run_id,
                    "bundle_dir": str(bundle_dir.resolve()),
                    "forecast_mode": forecast_mode,
                    "skip_reason": f"unsupported forecast_mode for replay benchmark: {forecast_mode}",
                }
            )
            continue

        runtime = build_forecast_runtime(
            historical_csv=historical_csv,
            current_dir=bundle_dir,
            artifacts_root=artifacts_root_path,
            dataset_key=dataset_key,
            catalog_path=catalog_path,
            weather_csv=weather_csv,
            holidays_xlsx=holidays_xlsx,
            device_request=device,
            benchmark_csv=benchmark_csv,
            allow_fallback=allow_fallback,
            logger=logger,
        )
        requested_run_id = (runtime.bundle.summary or {}).get("run_id", bundle_dir.name)
        model_metadata = _build_model_metadata(runtime, bundle_dir)
        replay = replay_forecast_period(runtime, start_date, end_date, logger=logger)
        replay_df = replay.replay_df
        skipped_days = replay.skipped_days
        model_dir = ensure_dir(out_dir / requested_run_id)
        replay_csv = model_dir / "replay_forecasts.csv"
        replay_df.to_csv(replay_csv, index=False)

        overall_metrics = evaluate_forecast_frame(replay_df)
        per_day_metrics = []
        per_day_model_usage = []
        if not replay_df.empty:
            for target_day, day_df in replay_df.groupby("target_date", dropna=False):
                day_metrics = evaluate_forecast_frame(day_df)
                if day_metrics is not None:
                    per_day_metrics.append({"target_date": str(target_day), **day_metrics})
                per_day_model_usage.append(
                    {
                        "target_date": str(target_day),
                        "model_run_ids": sorted(
                            str(x) for x in day_df["model_run_id"].dropna().unique().tolist()
                        ),
                    }
                )

        metrics_payload = {
            "requested_model_run_id": requested_run_id,
            "effective_model_run_ids": replay.effective_model_run_ids,
            "fallback_enabled": bool(allow_fallback),
            "fallback_used": replay.fallback_used,
            "n_requested_days": int(len(all_days)),
            "n_forecasted_days": int(len(all_days) - len(skipped_days)),
            "n_skipped_days": int(len(skipped_days)),
            "skipped_days": skipped_days,
            "start_date": start_date,
            "end_date": end_date,
            "n_days": int(len(all_days)),
            "n_rows": int(len(replay_df)),
            "dataset_key": runtime.data_config.get("dataset_key"),
            "forecast_mode": runtime.forecast_mode,
            "overall_metrics": overall_metrics,
            "per_day_metrics": per_day_metrics,
            "per_day_model_usage": per_day_model_usage,
            "replay_csv": str(replay_csv.resolve()),
            **model_metadata,
        }
        metrics_json = model_dir / "replay_metrics.json"
        save_json(metrics_json, metrics_payload)

        summary_rows.append(
            {
                "requested_model_run_id": requested_run_id,
                "effective_model_run_ids": "|".join(replay.effective_model_run_ids),
                "fallback_enabled": bool(allow_fallback),
                "fallback_used": replay.fallback_used,
                "n_requested_days": int(len(all_days)),
                "n_forecasted_days": int(len(all_days) - len(skipped_days)),
                "n_skipped_days": int(len(skipped_days)),
                "start_date": start_date,
                "end_date": end_date,
                "n_days": int(len(all_days)),
                "n_rows": int(len(replay_df)),
                "replay_csv": str(replay_csv.resolve()),
                "metrics_json": str(metrics_json.resolve()),
                **model_metadata,
                **(overall_metrics or {}),
            }
        )
        manifest["models"].append(
            {
                "requested_model_run_id": requested_run_id,
                "replay_csv": str(replay_csv.resolve()),
                "metrics_json": str(metrics_json.resolve()),
                **model_metadata,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        sort_cols = [col for col in ["MAE", "RMSE"] if col in summary_df.columns]
        if sort_cols:
            summary_df = summary_df.sort_values(sort_cols, ascending=True)
    summary_csv = out_dir / "replay_benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    manifest_path = out_dir / "replay_benchmark_manifest.json"
    save_json(manifest_path, manifest)

    return {
        "summary_csv": str(summary_csv.resolve()),
        "manifest_json": str(manifest_path.resolve()),
        "n_models": int(len(summary_df)),
        "start_date": start_date,
        "end_date": end_date,
        "rows": _rows_to_builtin(summary_df),
    }


def run_consumption_feature_benchmark(
    *,
    configs: list[str],
    output_csv: str = "artifacts/benchmarks/consumption_feature_variants.csv",
    analysis_days: int = 1,
    dataset_key: str | None = None,
    catalog_path: str | None = None,
    historical_csv: str | None = None,
    weather_csv: str | None = None,
    holidays_xlsx: str | None = None,
    benchmark_csv: str | None = None,
    replay_start_date: str | None = None,
    replay_end_date: str | None = None,
) -> dict[str, Any]:
    rows = []
    per_config_runs = []

    for config_path in configs:
        training_payload = run_consumption_training(
            config=config_path,
            analysis_days=analysis_days,
            dataset_key=dataset_key,
            catalog_path=catalog_path,
            historical_csv=historical_csv,
            weather_csv=weather_csv,
            holidays_xlsx=holidays_xlsx,
            benchmark_csv=benchmark_csv,
            promote=False,
            profile=False,
        )
        replay_start = replay_start_date or str(pd.Timestamp(training_payload["test_date_min"]).date())
        replay_end = replay_end_date or str(pd.Timestamp(training_payload["test_date_max"]).date())
        replay_payload = run_consumption_replay_benchmark(
            model_refs=[training_payload["run_id"]],
            start_date=replay_start,
            end_date=replay_end,
            dataset_key=dataset_key,
            catalog_path=catalog_path,
            historical_csv=historical_csv,
            weather_csv=weather_csv,
            holidays_xlsx=holidays_xlsx,
            benchmark_csv=benchmark_csv,
        )
        replay_rows = replay_payload["rows"]
        if not replay_rows:
            raise ApiServiceError(
                f"Replay benchmark returned no rows for config {config_path}.",
                status_code=500,
            )
        replay_row = replay_rows[0]
        replay_metric_keys = ["MAE", "RMSE", "Bias(ME)", "MAPE%", "InTolerance%", "count"]
        replay_metrics = {f"replay_{key}": replay_row.get(key) for key in replay_metric_keys}
        replay_metadata = {
            key: replay_row.get(key)
            for key in replay_row.keys()
            if key not in {"config_path", "feature_config", "feature_columns", *replay_metric_keys}
        }

        row = {
            "config": config_path,
            "run_id": training_payload["run_id"],
            "experiment_name": training_payload.get("experiment_name"),
            "dataset_key": training_payload.get("dataset_key"),
            "forecast_mode": training_payload.get("forecast_mode"),
            "selected_analysis_day": training_payload["selected_analysis_day"],
            "replay_start_date": replay_start,
            "replay_end_date": replay_end,
            "train_duration_sec": training_payload.get("train_duration_sec"),
            "n_features": training_payload.get("n_features"),
            "epochs_ran": training_payload.get("epochs_ran"),
            "best_val_loss": training_payload.get("best_val_loss"),
            "final_train_loss": training_payload.get("final_train_loss"),
            "final_val_loss": training_payload.get("final_val_loss"),
            "n_train_rows": training_payload.get("n_train_rows"),
            "n_val_rows": training_payload.get("n_val_rows"),
            "n_test_rows": training_payload.get("n_test_rows"),
            "feature_config_json": json.dumps(training_payload.get("feature_config"), sort_keys=True),
            "classical_MAE": training_payload["metrics_model"]["MAE"],
            "classical_RMSE": training_payload["metrics_model"]["RMSE"],
            "replay_summary_csv": replay_payload["summary_csv"],
            "replay_metrics_json": replay_row.get("metrics_json"),
            **replay_metrics,
            **replay_metadata,
        }
        rows.append(row)
        per_config_runs.append(
            {
                "config": config_path,
                "run_id": training_payload["run_id"],
                "replay_summary_csv": replay_payload["summary_csv"],
                "replay_manifest_json": replay_payload["manifest_json"],
            }
        )

    df = pd.DataFrame(rows)
    output_path = Path(output_csv)
    ensure_dir(output_path.parent)
    sort_cols = [col for col in ["replay_MAE", "replay_RMSE"] if col in df.columns]
    sorted_df = df.sort_values(by=sort_cols) if sort_cols else df
    sorted_df.to_csv(output_path, index=False)

    return {
        "output_csv": str(output_path.resolve()),
        "n_configs": len(configs),
        "rows": _rows_to_builtin(sorted_df),
        "runs": _to_builtin(per_config_runs),
    }
