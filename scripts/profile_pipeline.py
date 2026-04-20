from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from smartgrid.common.logging import build_log_path, setup_logger
from smartgrid.common.profiling import (
    build_environment_summary,
    build_runtime_diagnostics,
    write_json_report,
)
from smartgrid.common.utils import ensure_dir, load_yaml
from smartgrid.data.catalog import resolve_consumption_data_config
from smartgrid.inference.day_ahead import (
    build_forecast_runtime,
    profile_forecast_target_day,
    profile_replay_forecast_period,
    write_forecast_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the SmartGrid train/predict/replay pipeline")
    parser.add_argument("--config", default="configs/consumption/mlp_baseline.yaml")
    parser.add_argument("--analysis-days", type=int, default=1)
    parser.add_argument("--predict-target-date", default="2026-01-15")
    parser.add_argument("--replay-start-date", default="2026-01-01")
    parser.add_argument("--replay-end-date", default="2026-01-07")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--current-dir", default="artifacts/models/consumption/current")
    parser.add_argument("--dataset-key", default=None)
    parser.add_argument("--catalog-path", default=None)
    parser.add_argument("--historical-csv", default=None)
    parser.add_argument("--weather-csv", default=None)
    parser.add_argument("--holidays-xlsx", default=None)
    parser.add_argument("--benchmark-csv", default=None)
    parser.add_argument("--allow-fallback", action="store_true")
    return parser.parse_args()


def run_training(args: argparse.Namespace, logger) -> tuple[dict, float, list[str]]:
    command = [
        sys.executable,
        "scripts/train_consumption.py",
        "--config",
        args.config,
        "--analysis-days",
        str(args.analysis_days),
        "--promote",
        "--profile",
        "--device",
        args.device,
    ]
    if args.dataset_key:
        command.extend(["--dataset-key", args.dataset_key])
    if args.catalog_path:
        command.extend(["--catalog-path", args.catalog_path])
    if args.historical_csv:
        command.extend(["--historical-csv", args.historical_csv])
    if args.weather_csv:
        command.extend(["--weather-csv", args.weather_csv])
    if args.holidays_xlsx:
        command.extend(["--holidays-xlsx", args.holidays_xlsx])
    if args.benchmark_csv:
        command.extend(["--benchmark-csv", args.benchmark_csv])

    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    wall_time_sec = time.perf_counter() - start
    payload = json.loads(result.stdout)
    summary_path = Path(payload["exports_dir"]) / "run_summary.json"
    training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    logger.info("Training command finished wall_time_sec=%.3f run_id=%s", wall_time_sec, payload["run_id"])
    return training_summary, wall_time_sec, command


def render_markdown_report(report: dict) -> str:
    env = report["environment"]
    training = report["training"]
    prediction = report["prediction"]
    replay = report["replay"]
    micro = training["trainer"]["batch_micro_average_sec"]
    pipeline = training["pipeline_timings_sec"]
    training_total = training["wall_time_sec"]

    rows = []
    ordered = [
        ("data_loading_sec", "data loading"),
        ("holiday_weather_load_sec", "holiday/weather loading and merge"),
        ("feature_engineering_sec", "feature engineering"),
        ("split_scaling_prep_sec", "split/scaling/prep"),
        ("dataloader_tensor_creation_sec", "dataloader/tensor creation"),
        ("train_loop_total_sec", "train loop total"),
        ("validation_total_sec", "validation total"),
        ("post_train_evaluation_sec", "post-train evaluation"),
        ("export_artifact_writing_sec", "export / artifact writing"),
    ]
    for key, label in ordered:
        value = pipeline.get(key, 0.0)
        pct = (value / training_total * 100.0) if training_total else 0.0
        rows.append(f"| {label} | {value:.3f} | {pct:.1f}% |")

    slow_days = sorted(replay["per_day_sec"], key=lambda item: item["elapsed_sec"], reverse=True)
    slowest_day = slow_days[0] if slow_days else None

    bottlenecks = report["bottlenecks"]
    bottleneck_lines = [f"{idx}. {item}" for idx, item in enumerate(bottlenecks, start=1)]

    return "\n".join(
        [
            "# SmartGrid Profiling Report",
            "",
            "## A. Environment summary",
            "",
            f"- Python version: `{env['python_version']}`",
            f"- PyTorch version: `{env['pytorch_version']}`",
            f"- CUDA availability: `{env['cuda_available']}`",
            f"- Selected device: `{env['selected_device']}`",
            f"- GPU name: `{env['gpu_name']}`",
            f"- Config path used: `{env['config_path']}`",
            f"- Dataset key: `{env['dataset_key']}`",
            f"- Historical CSV: `{env['historical_csv']}`",
            f"- Weather CSV: `{env['weather_csv']}`",
            f"- Holidays XLSX: `{env['holidays_xlsx']}`",
            "",
            "## B. Training pipeline timing summary",
            "",
            f"- Total training script wall time: `{training_total:.3f}` sec",
            "",
            "| Stage | Seconds | Share of total |",
            "| --- | ---: | ---: |",
            *rows,
            "",
            "## C. Per-epoch training summary",
            "",
            f"- Epochs run: `{training['trainer']['epochs_ran']}`",
            f"- Average epoch duration: `{training['trainer']['epoch_duration_sec']['avg']:.3f}` sec",
            f"- Min epoch duration: `{training['trainer']['epoch_duration_sec']['min']:.3f}` sec",
            f"- Max epoch duration: `{training['trainer']['epoch_duration_sec']['max']:.3f}` sec",
            f"- Best validation loss epoch: `{training['trainer']['best_val_loss_epoch']}`",
            "",
            "## D. Per-batch micro-breakdown",
            "",
            f"- Sampled batches: `{micro['samples']}`",
            f"- Batch fetch wait: `{micro['batch_wait_sec']:.6f}` sec",
            f"- H2D copy: `{micro['h2d_sec']:.6f}` sec",
            f"- Forward: `{micro['forward_sec']:.6f}` sec",
            f"- Backward: `{micro['backward_sec']:.6f}` sec",
            f"- Optimizer step: `{micro['optimizer_sec']:.6f}` sec",
            f"- Metrics/logging: `{micro['metrics_sec']:.6f}` sec",
            "",
            "## E. Prediction timing summary",
            "",
            f"- Total prediction wall time: `{prediction['total_wall_time_sec']:.3f}` sec",
            f"- Runtime/model loading: `{prediction['runtime_build_sec']:.3f}` sec",
            f"- History/data loading: `{prediction['history_loading_sec']:.3f}` sec",
            f"- Target-day feature preparation: `{prediction['target_day_feature_preparation_sec']:.3f}` sec",
            f"- Forecast loop: `{prediction['forecast_loop_sec']:.3f}` sec",
            f"- Output writing: `{prediction['output_writing_sec']:.3f}` sec",
            "",
            "## F. Replay timing summary",
            "",
            f"- Total replay time: `{replay['total_replay_sec']:.3f}` sec",
            f"- Number of days requested: `{replay['n_requested_days']}`",
            f"- Number of days actually replayed: `{replay['n_forecasted_days']}`",
            f"- Average time per day: `{replay['avg_time_per_day_sec']:.3f}` sec",
            f"- Slowest day: `{slowest_day['target_date'] if slowest_day else None}` ({slowest_day['elapsed_sec']:.3f} sec)" if slowest_day else "- Slowest day: `None`",
            "",
            "## G. Bottleneck conclusion",
            "",
            *bottleneck_lines,
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bench_dir = ensure_dir(Path(args.artifacts_root) / "benchmarks" / stamp)
    if args.device == "cpu":
        wrapper_selected_device = torch.device("cpu")
    elif args.device == "cuda":
        wrapper_selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        wrapper_selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(
        "smartgrid.profile",
        log_file=build_log_path(args.artifacts_root, "benchmarks", f"{stamp}.log"),
    )
    wrapper_runtime_diagnostics = build_runtime_diagnostics(
        requested_device=args.device,
        selected_device=wrapper_selected_device,
        profiling_enabled=True,
    )
    logger.info("Wrapper runtime diagnostics=%s", wrapper_runtime_diagnostics)

    training_summary, training_wall_time_sec, train_command = run_training(args, logger)

    config = load_yaml(args.config)
    data_config = resolve_consumption_data_config(
        config["data"],
        dataset_key=args.dataset_key,
        catalog_path=args.catalog_path,
        overrides={
            "historical_csv": args.historical_csv,
            "weather_csv": args.weather_csv,
            "holidays_xlsx": args.holidays_xlsx,
            "benchmark_csv": args.benchmark_csv,
        },
    )

    runtime_build_start = time.perf_counter()
    runtime = build_forecast_runtime(
        historical_csv=args.historical_csv,
        current_dir=args.current_dir,
        artifacts_root=args.artifacts_root,
        dataset_key=args.dataset_key,
        catalog_path=args.catalog_path,
        weather_csv=args.weather_csv,
        holidays_xlsx=args.holidays_xlsx,
        device_request=args.device,
        benchmark_csv=args.benchmark_csv,
        allow_fallback=args.allow_fallback,
        logger=logger,
    )
    runtime_build_sec = time.perf_counter() - runtime_build_start

    prediction_start = time.perf_counter()
    forecast_df, forecast_profile = profile_forecast_target_day(
        runtime,
        args.predict_target_date,
        logger=logger,
    )
    write_start = time.perf_counter()
    output_paths = write_forecast_outputs(
        forecast_df,
        runtime.artifacts_root,
        forecast_profile.target_date,
        forecast_profile.model_run_id,
    )
    output_writing_sec = time.perf_counter() - write_start
    prediction_flow_sec = time.perf_counter() - prediction_start
    prediction_total_sec = runtime_build_sec + prediction_flow_sec

    replay_profile = profile_replay_forecast_period(
        runtime,
        args.replay_start_date,
        args.replay_end_date,
        logger=logger,
    )
    replay_summary = replay_profile.summary
    replay_requested_days = len(replay_profile.per_day_sec)
    replay_forecasted_days = sum(1 for item in replay_profile.per_day_sec if item["status"] == "ok")
    replay_avg_time_sec = (
        sum(float(item["elapsed_sec"]) for item in replay_profile.per_day_sec) / replay_requested_days
        if replay_requested_days
        else 0.0
    )

    pipeline_timings = dict(training_summary.get("profiling", {}).get("pipeline_timings_sec", {}))
    bottleneck_pairs = sorted(
        (
            ("training:" + key, float(value))
            for key, value in pipeline_timings.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    bottleneck_pairs.extend(
        [
            ("prediction:runtime_build_sec", runtime_build_sec),
            ("prediction:forecast_loop_sec", forecast_profile.timings_sec.get("forecast_loop_sec", 0.0)),
            ("replay:total_replay_sec", replay_profile.total_replay_sec),
        ]
    )
    bottleneck_pairs = sorted(bottleneck_pairs, key=lambda item: item[1], reverse=True)
    bottlenecks = [f"{name} at {value:.3f} sec" for name, value in bottleneck_pairs[:4]]

    environment = build_environment_summary(runtime.device, args.config, data_config)
    report = {
        "run_id": stamp,
        "environment": environment,
        "wrapper_runtime_diagnostics": wrapper_runtime_diagnostics,
        "commands": {
            "profile": sys.argv,
            "train": train_command,
        },
        "training": {
            "run_id": training_summary["run_id"],
            "wall_time_sec": training_wall_time_sec,
            "pipeline_timings_sec": pipeline_timings,
            "trainer": training_summary.get("profiling", {}).get("trainer", {}),
            "summary_path": training_summary["run_dir"] + "/run_summary.json",
            "exports_dir": training_summary["exports_dir"],
        },
        "prediction": {
            "target_date": forecast_profile.target_date,
            "runtime_build_sec": runtime_build_sec,
            "history_loading_sec": runtime_build_sec,
            "target_day_feature_preparation_sec": forecast_profile.timings_sec.get("target_day_feature_preparation_sec", 0.0),
            "forecast_loop_sec": forecast_profile.timings_sec.get("forecast_loop_sec", 0.0),
            "model_inference_sec": forecast_profile.timings_sec.get("model_inference_sec", 0.0),
            "output_writing_sec": output_writing_sec,
            "prediction_flow_sec": prediction_flow_sec,
            "total_wall_time_sec": prediction_total_sec,
            "current_output_csv": str(output_paths.current_output_path.resolve()),
            "archive_output_csv": str(output_paths.archive_output_path.resolve()),
        },
        "replay": {
            "start_date": replay_summary.start_date,
            "end_date": replay_summary.end_date,
            "total_replay_sec": replay_profile.total_replay_sec,
            "n_requested_days": replay_requested_days,
            "n_forecasted_days": replay_forecasted_days,
            "n_skipped_days": len(replay_summary.skipped_days),
            "avg_time_per_day_sec": replay_avg_time_sec,
            "skipped_days": replay_summary.skipped_days,
            "per_day_sec": replay_profile.per_day_sec,
        },
        "artifacts": {
            "benchmark_dir": str(bench_dir.resolve()),
            "training_run_dir": training_summary["run_dir"],
            "training_exports_dir": training_summary["exports_dir"],
            "prediction_current_output_csv": str(output_paths.current_output_path.resolve()),
            "prediction_archive_output_csv": str(output_paths.archive_output_path.resolve()),
        },
        "bottlenecks": bottlenecks,
    }

    report["paths"] = {
        "json_report": str((bench_dir / "profile_report.json").resolve()),
        "markdown_report": str((bench_dir / "profile_report.md").resolve()),
    }
    write_json_report(bench_dir / "profile_report.json", report)
    (bench_dir / "profile_report.md").write_text(render_markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
