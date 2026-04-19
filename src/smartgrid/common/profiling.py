from __future__ import annotations

import json
import os
import platform
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from smartgrid.common.utils import ensure_dir


def maybe_cuda_synchronize(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@contextmanager
def timed_block(
    timings: dict[str, float],
    key: str,
    *,
    device: torch.device | None = None,
) -> Any:
    maybe_cuda_synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        maybe_cuda_synchronize(device)
        timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start)


@dataclass(slots=True)
class BatchTimingAggregate:
    samples: int = 0
    batch_wait_sec: float = 0.0
    h2d_sec: float = 0.0
    forward_sec: float = 0.0
    backward_sec: float = 0.0
    optimizer_sec: float = 0.0
    metrics_sec: float = 0.0

    def add_sample(
        self,
        *,
        batch_wait_sec: float,
        h2d_sec: float,
        forward_sec: float,
        backward_sec: float,
        optimizer_sec: float,
        metrics_sec: float,
    ) -> None:
        self.samples += 1
        self.batch_wait_sec += batch_wait_sec
        self.h2d_sec += h2d_sec
        self.forward_sec += forward_sec
        self.backward_sec += backward_sec
        self.optimizer_sec += optimizer_sec
        self.metrics_sec += metrics_sec

    def average_dict(self) -> dict[str, float]:
        if self.samples == 0:
            return {
                "samples": 0,
                "batch_wait_sec": 0.0,
                "h2d_sec": 0.0,
                "forward_sec": 0.0,
                "backward_sec": 0.0,
                "optimizer_sec": 0.0,
                "metrics_sec": 0.0,
            }
        return {
            "samples": self.samples,
            "batch_wait_sec": self.batch_wait_sec / self.samples,
            "h2d_sec": self.h2d_sec / self.samples,
            "forward_sec": self.forward_sec / self.samples,
            "backward_sec": self.backward_sec / self.samples,
            "optimizer_sec": self.optimizer_sec / self.samples,
            "metrics_sec": self.metrics_sec / self.samples,
        }


@dataclass(slots=True)
class TrainerProfiler:
    enabled: bool = False
    epoch_durations_sec: list[float] = field(default_factory=list)
    train_loop_total_sec: float = 0.0
    validation_loop_total_sec: float = 0.0
    batch_timings: BatchTimingAggregate = field(default_factory=BatchTimingAggregate)

    def record_epoch_duration(self, value: float) -> None:
        if self.enabled:
            self.epoch_durations_sec.append(value)

    def to_summary(self, history: dict[str, list[float]] | None = None) -> dict[str, Any]:
        epochs_ran = len(self.epoch_durations_sec)
        best_epoch = None
        if history and history.get("val_loss"):
            best_epoch = int(min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__) + 1)
        return {
            "enabled": self.enabled,
            "epochs_ran": epochs_ran,
            "epoch_duration_sec": {
                "avg": statistics.fmean(self.epoch_durations_sec) if self.epoch_durations_sec else 0.0,
                "min": min(self.epoch_durations_sec) if self.epoch_durations_sec else 0.0,
                "max": max(self.epoch_durations_sec) if self.epoch_durations_sec else 0.0,
                "values": list(self.epoch_durations_sec),
            },
            "train_loop_total_sec": self.train_loop_total_sec,
            "validation_loop_total_sec": self.validation_loop_total_sec,
            "batch_micro_average_sec": self.batch_timings.average_dict(),
            "best_val_loss_epoch": best_epoch,
        }


def build_environment_summary(device: torch.device, config_path: str, data_config: dict) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else None
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "selected_device": str(device),
        "gpu_name": gpu_name,
        "config_path": str(Path(config_path).resolve()),
        "dataset_key": data_config.get("dataset_key"),
        "historical_csv": str(Path(data_config["historical_csv"]).resolve()) if data_config.get("historical_csv") else None,
        "weather_csv": str(Path(data_config["weather_csv"]).resolve()) if data_config.get("weather_csv") else None,
        "holidays_xlsx": str(Path(data_config["holidays_xlsx"]).resolve()) if data_config.get("holidays_xlsx") else None,
        "benchmark_csv": str(Path(data_config["benchmark_csv"]).resolve()) if data_config.get("benchmark_csv") else None,
    }


def build_runtime_diagnostics(*, requested_device: str, selected_device: torch.device, profiling_enabled: bool) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpu_names = [torch.cuda.get_device_name(idx) for idx in range(device_count)] if cuda_available else []
    return {
        "executable": str(Path(os.sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_device_names": gpu_names,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requested_device": requested_device,
        "selected_device": str(selected_device),
        "profiling_enabled": bool(profiling_enabled),
    }


def write_json_report(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
