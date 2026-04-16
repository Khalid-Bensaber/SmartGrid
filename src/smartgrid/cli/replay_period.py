from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from smartgrid.common.logging import build_log_path, setup_logger
from smartgrid.common.paths import build_replay_paths
from smartgrid.evaluation.reporting import evaluate_forecast_frame, save_json
from smartgrid.inference.day_ahead import build_forecast_runtime, replay_forecast_period


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a historical period day by day without leakage"
    )
    parser.add_argument("--historical-csv", default=None)
    parser.add_argument("--dataset-key", default=None)
    parser.add_argument("--catalog-path", default=None)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--weather-csv", default=None)
    parser.add_argument("--holidays-xlsx", default=None)
    parser.add_argument("--current-dir", default="artifacts/models/consumption/current")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--benchmark-csv", default=None)
    parser.add_argument("--write-per-day", action="store_true")
    parser.add_argument("--allow-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger = setup_logger(
        "smartgrid.replay",
        log_file=build_log_path(
            args.artifacts_root,
            "replay",
            f"{stamp}__{args.start_date}__{args.end_date}.log",
        ),
    )

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

    replay_paths = build_replay_paths(args.artifacts_root, args.start_date, args.end_date, stamp)
    replay = replay_forecast_period(runtime, args.start_date, args.end_date, logger=logger)
    replay_df = replay.replay_df
    all_days = pd.date_range(replay.start_date, replay.end_date, freq="D")
    skipped_days = replay.skipped_days
    logger.info(
        "Starting replay from %s to %s total_days=%s",
        replay.start_date,
        replay.end_date,
        len(all_days),
    )

    if args.write_per_day:
        for target_date, day_forecast in replay_df.groupby("target_date", dropna=False):
            day_path = replay_paths.per_day_dir / f"forecast_{target_date}.csv"
            day_forecast.to_csv(day_path, index=False)

    replay_df.to_csv(replay_paths.output_csv, index=False)

    overall_metrics = evaluate_forecast_frame(replay_df)
    per_day_metrics = []
    model_usage = []
    if not replay_df.empty:
        for target_date, day_df in replay_df.groupby("target_date", dropna=False):
            metrics = evaluate_forecast_frame(day_df)
            if metrics is not None:
                per_day_metrics.append({"target_date": target_date, **metrics})
            model_ids = sorted(str(x) for x in day_df["model_run_id"].dropna().unique().tolist())
            model_usage.append(
                {
                    "target_date": target_date,
                    "model_run_ids": model_ids,
                    "fallback_used": any(
                        model_id != (runtime.bundle.summary or {}).get("run_id", "unknown")
                        for model_id in model_ids
                    ),
                }
            )

    requested_model_run_id = replay.requested_model_run_id
    effective_model_run_ids = replay.effective_model_run_ids
    metrics_payload = {
        "start_date": replay.start_date,
        "end_date": replay.end_date,
        "n_days": int(len(all_days)),
        "n_rows": int(len(replay_df)),
        "requested_model_run_id": requested_model_run_id,
        "effective_model_run_ids": effective_model_run_ids,
        "fallback_enabled": bool(args.allow_fallback),
        "fallback_used": replay.fallback_used,
        "n_requested_days": int(len(all_days)),
        "n_forecasted_days": int(len(all_days) - len(skipped_days)),
        "n_skipped_days": int(len(skipped_days)),
        "skipped_days": skipped_days,
        "dataset_key": runtime.data_config.get("dataset_key"),
        "forecast_mode": runtime.forecast_mode,
        "overall_metrics": overall_metrics,
        "per_day_metrics": per_day_metrics,
        "per_day_model_usage": model_usage,
        "output_csv": str(replay_paths.output_csv.resolve()),
    }
    save_json(replay_paths.metrics_json, metrics_payload)
    logger.info(
        "Replay outputs written csv=%s metrics=%s",
        replay_paths.output_csv,
        replay_paths.metrics_json,
    )
    print(
        json.dumps(
            {
                "start_date": replay.start_date,
                "end_date": replay.end_date,
                "n_days": int(len(all_days)),
                "n_rows": int(len(replay_df)),
                "requested_model_run_id": requested_model_run_id,
                "effective_model_run_ids": effective_model_run_ids,
                "fallback_used": replay.fallback_used,
                "n_requested_days": int(len(all_days)),
                "n_forecasted_days": int(len(all_days) - len(skipped_days)),
                "n_skipped_days": int(len(skipped_days)),
                "dataset_key": runtime.data_config.get("dataset_key"),
                "forecast_mode": runtime.forecast_mode,
                "output_csv": str(replay_paths.output_csv.resolve()),
                "metrics_json": str(replay_paths.metrics_json.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
