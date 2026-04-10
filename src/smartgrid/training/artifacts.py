from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import torch
from torch import nn

from smartgrid.common.utils import ensure_dir


def save_training_bundle(model: nn.Module, x_scaler, y_scaler, run_dir: Path, model_config: dict, run_summary: dict | None = None) -> dict[str, Path]:
    run_dir = ensure_dir(run_dir)
    model_path = run_dir / "model.pt"
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": model_config,
    }
    torch.save(checkpoint, model_path)

    x_scaler_path = run_dir / "x_scaler.pkl"
    y_scaler_path = run_dir / "y_scaler.pkl"
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)

    outputs = {
        "model": model_path,
        "x_scaler": x_scaler_path,
        "y_scaler": y_scaler_path,
    }
    if run_summary is not None:
        summary_path = run_dir / "run_summary.json"
        summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
        outputs["summary"] = summary_path
    return outputs


def promote_bundle(run_dir: Path, current_dir: Path) -> Path:
    current_dir = ensure_dir(current_dir)
    for child in current_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            shutil.rmtree(child)

    for src in run_dir.iterdir():
        dst = current_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return current_dir
