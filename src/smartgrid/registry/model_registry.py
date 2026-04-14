from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch

from smartgrid.common.utils import load_yaml
from smartgrid.models.mlp import TorchMLP


@dataclass(slots=True)
class ConsumptionBundle:
    model: TorchMLP
    x_scaler: Any
    y_scaler: Any
    model_config: dict
    summary: dict | None
    training_config: dict | None


def load_current_consumption_bundle(current_dir: str | Path, device: torch.device) -> ConsumptionBundle:
    current_dir = Path(current_dir)
    checkpoint = torch.load(current_dir / "model.pt", map_location=device)
    model_config = checkpoint["model_config"]
    model = TorchMLP(
        input_dim=model_config["input_dim"],
        hidden_layers=model_config["hidden_layers"],
        dropout=model_config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x_scaler = joblib.load(current_dir / "x_scaler.pkl")
    y_scaler = joblib.load(current_dir / "y_scaler.pkl")
    summary_path = current_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None

    training_config = None
    if summary is not None and summary.get("config_path"):
        config_path = Path(summary["config_path"])
        if config_path.exists():
            training_config = load_yaml(config_path)

    return ConsumptionBundle(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        model_config=model_config,
        summary=summary,
        training_config=training_config,
    )
