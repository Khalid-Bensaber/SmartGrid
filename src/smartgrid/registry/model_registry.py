from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch

from smartgrid.common.utils import load_yaml
from smartgrid.models.mlp import TorchMLP


@dataclass(slots=True)
class ConsumptionBundle:
    bundle_dir: Path
    model: TorchMLP
    x_scaler: Any
    y_scaler: Any
    model_config: dict
    summary: dict | None
    training_config: dict | None


def load_consumption_bundle(bundle_dir: str | Path, device: torch.device) -> ConsumptionBundle:
    bundle_dir = Path(bundle_dir)
    checkpoint = torch.load(bundle_dir / "model.pt", map_location=device)
    model_config = checkpoint["model_config"]
    model = TorchMLP(
        input_dim=model_config["input_dim"],
        hidden_layers=model_config["hidden_layers"],
        dropout=model_config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x_scaler = joblib.load(bundle_dir / "x_scaler.pkl")
    y_scaler = joblib.load(bundle_dir / "y_scaler.pkl")
    summary_path = bundle_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None

    training_config = None
    if summary is not None and summary.get("config_path"):
        config_path = Path(summary["config_path"])
        if config_path.exists():
            training_config = load_yaml(config_path)

    return ConsumptionBundle(
        bundle_dir=bundle_dir,
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        model_config=model_config,
        summary=summary,
        training_config=training_config,
    )


def load_current_consumption_bundle(current_dir: str | Path, device: torch.device) -> ConsumptionBundle:
    return load_consumption_bundle(current_dir, device)


def list_ranked_consumption_bundle_dirs(
    *,
    current_dir: str | Path = "artifacts/models/consumption/current",
    runs_dir: str | Path = "artifacts/runs/consumption",
    benchmark_csv: str | Path | None = "artifacts/benchmarks/consumption_feature_variants.csv",
) -> list[Path]:
    current_dir = Path(current_dir)
    runs_dir = Path(runs_dir)
    ranked_dirs: list[Path] = []
    seen: set[Path] = set()

    if current_dir.exists() and (current_dir / "model.pt").exists():
        ranked_dirs.append(current_dir)
        seen.add(current_dir.resolve())

    benchmark_path = Path(benchmark_csv) if benchmark_csv is not None else None
    if benchmark_path is not None and benchmark_path.exists():
        bench = pd.read_csv(benchmark_path)
        sort_cols = [
            col for col in ["replay_MAE", "replay_RMSE", "MAE", "RMSE"] if col in bench.columns
        ]
        if sort_cols:
            bench = bench.sort_values(sort_cols, ascending=True)
        for _, row in bench.iterrows():
            run_id = row.get("run_id")
            if pd.isna(run_id):
                continue
            run_dir = runs_dir / str(run_id)
            if run_dir.exists() and (run_dir / "model.pt").exists():
                resolved = run_dir.resolve()
                if resolved not in seen:
                    ranked_dirs.append(run_dir)
                    seen.add(resolved)

    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir() or not (run_dir / "model.pt").exists():
                continue
            resolved = run_dir.resolve()
            if resolved not in seen:
                ranked_dirs.append(run_dir)
                seen.add(resolved)

    return ranked_dirs
