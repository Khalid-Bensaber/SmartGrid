from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from smartgrid.common.logging import build_log_path, setup_logger
from smartgrid.common.utils import ensure_dir
from smartgrid.inference.day_ahead import (
    build_forecast_runtime,
    forecast_target_day,
    infer_target_date_from_history,
    write_forecast_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast one full target day using the promoted model only")
    parser.add_argument("--historical-csv", required=True)
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--weather-csv", default=None)
    parser.add_argument("--holidays-xlsx", default=None)
    parser.add_argument("--current-dir", default="artifacts/models/consumption/current")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    requested_target = args.target_date or "auto"
    logger = setup_logger(
        "smartgrid.predict",
        log_file=build_log_path(args.artifacts_root, "predict", f"{stamp}__forecast_{requested_target}.log"),
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
    target_date = args.target_date or infer_target_date_from_history(runtime.historical_df, runtime.date_col)
    logger.info("Forecast request target_date=%s requested=%s", target_date, requested_target)

    forecast_df = forecast_target_day(runtime, target_date, logger=logger)
    run_id = (runtime.bundle.summary or {}).get("run_id", "unknown")
    output_paths = write_forecast_outputs(forecast_df, runtime.artifacts_root, target_date, run_id)

    custom_output = None
    if args.output_csv is not None:
        custom_output = Path(args.output_csv)
        ensure_dir(custom_output.parent)
        forecast_df.to_csv(custom_output, index=False)

    logger.info(
        "Forecast outputs written current=%s archive=%s custom=%s",
        output_paths.current_output_path,
        output_paths.archive_output_path,
        custom_output,
    )
    print(
        json.dumps(
            {
                "target_date": target_date,
                "points": int(len(forecast_df)),
                "model_run_id": run_id,
                "current_output_csv": str(output_paths.current_output_path.resolve()),
                "archive_output_csv": str(output_paths.archive_output_path.resolve()),
                "custom_output_csv": str(custom_output.resolve()) if custom_output else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
