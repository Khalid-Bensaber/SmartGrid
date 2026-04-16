from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train several configs and rank them on strict replay metrics"
    )
    parser.add_argument("configs", nargs="+", help="List of YAML config paths")
    parser.add_argument(
        "--output-csv",
        default="artifacts/benchmarks/consumption_feature_variants.csv",
    )
    parser.add_argument("--analysis-days", type=int, default=1)
    parser.add_argument("--dataset-key", default=None)
    parser.add_argument("--catalog-path", default=None)
    parser.add_argument("--historical-csv", default=None)
    parser.add_argument("--weather-csv", default=None)
    parser.add_argument("--holidays-xlsx", default=None)
    parser.add_argument("--benchmark-csv", default=None)
    parser.add_argument("--replay-start-date", default=None)
    parser.add_argument("--replay-end-date", default=None)
    return parser.parse_args()


def extract_last_json(stdout: str) -> dict:
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(stdout) if ch == "{"]

    for start in reversed(starts):
        snippet = stdout[start:]
        try:
            payload, end = decoder.raw_decode(snippet)
            if snippet[end:].strip() == "":
                return payload
        except json.JSONDecodeError:
            continue

    raise RuntimeError("Could not find a valid final JSON summary in training output.")


def main() -> None:
    args = parse_args()
    rows = []
    for config_path in args.configs:
        train_cmd = [
            "python",
            "scripts/train_consumption.py",
            "--config",
            config_path,
            "--analysis-days",
            str(args.analysis_days),
        ]
        if args.dataset_key:
            train_cmd.extend(["--dataset-key", args.dataset_key])
        if args.catalog_path:
            train_cmd.extend(["--catalog-path", args.catalog_path])
        if args.historical_csv:
            train_cmd.extend(["--historical-csv", args.historical_csv])
        if args.weather_csv:
            train_cmd.extend(["--weather-csv", args.weather_csv])
        if args.holidays_xlsx:
            train_cmd.extend(["--holidays-xlsx", args.holidays_xlsx])
        if args.benchmark_csv:
            train_cmd.extend(["--benchmark-csv", args.benchmark_csv])

        print(f"[TRAIN] {' '.join(train_cmd)}")
        completed = subprocess.run(train_cmd, check=True, capture_output=True, text=True)
        if completed.stderr.strip():
            print(completed.stderr)
        print(completed.stdout)
        payload = extract_last_json(completed.stdout)
        replay_start_date = args.replay_start_date or str(
            pd.Timestamp(payload["test_date_min"]).date()
        )
        replay_end_date = args.replay_end_date or str(
            pd.Timestamp(payload["test_date_max"]).date()
        )

        replay_cmd = [
            "python",
            "scripts/benchmark_replay_models.py",
            "--start-date",
            replay_start_date,
            "--end-date",
            replay_end_date,
            payload["run_id"],
        ]
        if args.dataset_key:
            replay_cmd.extend(["--dataset-key", args.dataset_key])
        if args.catalog_path:
            replay_cmd.extend(["--catalog-path", args.catalog_path])
        if args.historical_csv:
            replay_cmd.extend(["--historical-csv", args.historical_csv])
        if args.weather_csv:
            replay_cmd.extend(["--weather-csv", args.weather_csv])
        if args.holidays_xlsx:
            replay_cmd.extend(["--holidays-xlsx", args.holidays_xlsx])
        if args.benchmark_csv:
            replay_cmd.extend(["--benchmark-csv", args.benchmark_csv])

        print(f"[REPLAY] {' '.join(replay_cmd)}")
        replay_completed = subprocess.run(replay_cmd, check=True, capture_output=True, text=True)
        if replay_completed.stderr.strip():
            print(replay_completed.stderr)
        print(replay_completed.stdout)
        replay_payload = extract_last_json(replay_completed.stdout)
        replay_summary = pd.read_csv(replay_payload["summary_csv"])
        if replay_summary.empty:
            raise RuntimeError(f"Replay benchmark returned no rows for config {config_path}.")
        replay_row = replay_summary.iloc[0].to_dict()
        replay_metadata = {
            key: replay_row.get(key)
            for key in replay_row.keys()
            if key not in {"config_path", "feature_config", "feature_columns"}
        }

        row = {
            "config": config_path,
            "run_id": payload["run_id"],
            "experiment_name": payload.get("experiment_name"),
            "dataset_key": payload.get("dataset_key"),
            "forecast_mode": payload.get("forecast_mode"),
            "selected_analysis_day": payload["selected_analysis_day"],
            "replay_start_date": replay_start_date,
            "replay_end_date": replay_end_date,
            "train_duration_sec": payload.get("train_duration_sec"),
            "n_features": payload.get("n_features"),
            "epochs_ran": payload.get("epochs_ran"),
            "best_val_loss": payload.get("best_val_loss"),
            "final_train_loss": payload.get("final_train_loss"),
            "final_val_loss": payload.get("final_val_loss"),
            "n_train_rows": payload.get("n_train_rows"),
            "n_val_rows": payload.get("n_val_rows"),
            "n_test_rows": payload.get("n_test_rows"),
            "feature_config_json": json.dumps(payload.get("feature_config"), sort_keys=True),
            "classical_MAE": payload["metrics_model"]["MAE"],
            "classical_RMSE": payload["metrics_model"]["RMSE"],
            "replay_summary_csv": replay_payload["summary_csv"],
            "replay_metrics_json": replay_row.get("metrics_json"),
            **replay_metadata,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(by=["MAE", "RMSE"]).to_csv(output_path, index=False)
    print(f"[DONE] Wrote benchmark table to {output_path}")
    print(df.sort_values(by=["MAE", "RMSE"]).to_string(index=False))


if __name__ == "__main__":
    main()
