from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run several feature configs and aggregate results")
    parser.add_argument("configs", nargs="+", help="List of YAML config paths")
    parser.add_argument("--output-csv", default="artifacts/benchmarks/consumption_feature_variants.csv")
    parser.add_argument("--analysis-days", type=int, default=1)
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
        cmd = [
            "python",
            "scripts/train_consumption.py",
            "--config",
            config_path,
            "--analysis-days",
            str(args.analysis_days),
        ]
        print(f"[RUN] {' '.join(cmd)}")
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if completed.stderr.strip():
            print(completed.stderr)
        print(completed.stdout)
        payload = extract_last_json(completed.stdout)
        row = {
            "config": config_path,
            "run_id": payload["run_id"],
            "experiment_name": payload.get("experiment_name"),
            "selected_analysis_day": payload["selected_analysis_day"],
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
            **payload["metrics_model"],
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
