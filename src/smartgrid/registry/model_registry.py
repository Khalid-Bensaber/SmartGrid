from __future__ import annotations

import json
from pathlib import Path

import joblib
import torch

from smartgrid.models.mlp import TorchMLP


def load_current_consumption_bundle(current_dir: str | Path, device: torch.device):
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
    return model, x_scaler, y_scaler, model_config, summary
