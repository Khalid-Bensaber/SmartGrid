from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from smartgrid.common.logging import build_log_path, setup_logger
from smartgrid.common.paths import build_replay_paths
from smartgrid.evaluation.reporting import evaluate_forecast_frame, save_json
from smartgrid.inference.day_ahead import build_forecast_runtime, forecast_target_day


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a historical period day by day without leakage")
    parser.add_argument("--historical-csv", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--weather-csv", default=None)
    parser.add_argument("--holidays-xlsx", default=None)
    parser.add_argument("--current-dir", default="artifacts/models/consumption/current")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--write-per-day", action="store_true")
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
        weather_csv=args.weather_csv,
        holidays_xlsx=args.holidays_xlsx,
        device_request=args.device,
        logger=logger,
    )

    replay_paths = build_replay_paths(args.artifacts_root, args.start_date, args.end_date, stamp)
    all_days = pd.date_range(args.start_date, args.end_date, freq="D")
    logger.info("Starting replay from %s to %s total_days=%s", args.start_date, args.end_date, len(all_days))

    day_frames = []
    for day in all_days:
        target_date = str(day.date())
        day_forecast = forecast_target_day(runtime, target_date, logger=logger)
        day_frames.append(day_forecast)
        if args.write_per_day:
            day_path = replay_paths.per_day_dir / f"forecast_{target_date}.csv"
            day_forecast.to_csv(day_path, index=False)

    replay_df = pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame()
    replay_df.to_csv(replay_paths.output_csv, index=False)

    overall_metrics = evaluate_forecast_frame(replay_df)
    per_day_metrics = []
    if not replay_df.empty:
        for target_date, day_df in replay_df.groupby("target_date", dropna=False):
            metrics = evaluate_forecast_frame(day_df)
            if metrics is not None:
                per_day_metrics.append({"target_date": target_date, **metrics})

    metrics_payload = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_days": int(len(all_days)),
        "n_rows": int(len(replay_df)),
        "overall_metrics": overall_metrics,
        "per_day_metrics": per_day_metrics,
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
                "start_date": args.start_date,
                "end_date": args.end_date,
                "n_days": int(len(all_days)),
                "n_rows": int(len(replay_df)),
                "output_csv": str(replay_paths.output_csv.resolve()),
                "metrics_json": str(replay_paths.metrics_json.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
