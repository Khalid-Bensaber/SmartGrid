from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_TRAINING_FLOAT_KEYS = {"learning_rate", "weight_decay", "dropout"}
_TRAINING_INT_KEYS = {"seed", "batch_size", "epochs", "patience", "num_workers", "max_cuda_resident_bytes"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_hidden_layers(raw: list[int] | tuple[int, ...] | str) -> tuple[int, ...]:
    if isinstance(raw, str):
        return tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    return tuple(int(x) for x in raw)


def get_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _coerce_config_number(value: Any, caster: type[float] | type[int]) -> Any:
    if isinstance(value, str):
        try:
            return caster(value)
        except ValueError:
            return value
    return value


def _normalize_training_config(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    training = data.get("training")
    if not isinstance(training, dict):
        return data
    for key in _TRAINING_FLOAT_KEYS:
        if key in training:
            training[key] = _coerce_config_number(training[key], float)
    for key in _TRAINING_INT_KEYS:
        if key in training:
            training[key] = _coerce_config_number(training[key], int)
    return data


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return _normalize_training_config(loaded)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utc_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"
