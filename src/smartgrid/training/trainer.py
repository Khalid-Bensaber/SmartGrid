from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from math import ceil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from smartgrid.common.profiling import TrainerProfiler, maybe_cuda_synchronize
from smartgrid.common.utils import set_seed
from smartgrid.models.mlp import TorchMLP


@dataclass(slots=True)
class TrainResult:
    model: nn.Module
    history: dict
    test_x: torch.Tensor
    model_config: dict
    loader_prep_sec: float = 0.0
    batching_strategy: str = "dataloader"
    resident_data_bytes: int = 0


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.element_size() * tensor.numel())


def _arrays_to_float_tensors(*arrays) -> list[torch.Tensor]:
    return [torch.tensor(array, dtype=torch.float32) for array in arrays]


def make_loaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    batch_size: int,
    num_workers: int,
    *,
    pin_memory: bool,
):
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_x = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_x


def should_use_device_resident_batches(
    device: torch.device,
    *,
    strategy: str,
    resident_data_bytes: int,
    max_resident_bytes: int,
) -> bool:
    if strategy == "dataloader":
        return False
    if strategy == "device_resident":
        return device.type == "cuda"
    return device.type == "cuda" and resident_data_bytes <= max_resident_bytes


def build_device_resident_tensors(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    *,
    device: torch.device,
):
    train_x, train_y, val_x, val_y, test_x = _arrays_to_float_tensors(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
    )
    resident_data_bytes = sum(_tensor_bytes(tensor) for tensor in (train_x, train_y, val_x, val_y, test_x))
    train_x = train_x.to(device, non_blocking=True)
    train_y = train_y.to(device, non_blocking=True)
    val_x = val_x.to(device, non_blocking=True)
    val_y = val_y.to(device, non_blocking=True)
    test_x = test_x.to(device, non_blocking=True)
    return train_x, train_y, val_x, val_y, test_x, resident_data_bytes


def _split_batch_starts(length: int, batch_size: int) -> range:
    return range(0, length, batch_size)


def _mean_batch_metric(total: torch.Tensor, batch_count: int) -> float:
    if batch_count == 0:
        return 0.0
    return float((total / batch_count).detach().cpu().item())


def train_mlp_regressor(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    hidden_layers: tuple[int, ...],
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    dropout: float,
    seed: int,
    num_workers: int,
    device: torch.device,
    resume_checkpoint: str | None = None,
    logger: logging.Logger | None = None,
    profiler: TrainerProfiler | None = None,
    batching_strategy: str = "auto",
    max_cuda_resident_bytes: int = 512 * 1024 * 1024,
) -> TrainResult:
    set_seed(seed)
    profiler = profiler or TrainerProfiler(enabled=False)

    loader_prep_start = time.perf_counter()
    resident_data_bytes = 0
    using_device_resident_batches = False
    if device.type == "cuda":
        resident_data_bytes = sum(
            np.asarray(array, dtype=np.float32).nbytes
            for array in (X_train, y_train, X_val, y_val, X_test)
        )
    using_device_resident_batches = should_use_device_resident_batches(
        device,
        strategy=batching_strategy,
        resident_data_bytes=resident_data_bytes,
        max_resident_bytes=max_cuda_resident_bytes,
    )
    if using_device_resident_batches:
        train_x_tensor, train_y_tensor, val_x_tensor, val_y_tensor, test_x, resident_data_bytes = build_device_resident_tensors(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            device=device,
        )
        train_loader = None
        val_loader = None
    else:
        train_loader, val_loader, test_x = make_loaders(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
    loader_prep_sec = time.perf_counter() - loader_prep_start

    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model_config = checkpoint["model_config"]
        model = TorchMLP(
            input_dim=model_config["input_dim"],
            hidden_layers=model_config["hidden_layers"],
            dropout=model_config.get("dropout", 0.0),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = TorchMLP(input_dim=X_train.shape[1], hidden_layers=hidden_layers, dropout=dropout).to(device)
        model_config = {
            "model_type": "torch_mlp",
            "input_dim": int(X_train.shape[1]),
            "hidden_layers": list(hidden_layers),
            "dropout": float(dropout),
        }

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    epochs_without_improvement = 0
    global_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_total = torch.zeros((), device=device)
        train_mae_total = torch.zeros((), device=device)
        train_batch_count = 0
        train_loop_start = time.perf_counter()
        if using_device_resident_batches:
            n_train = int(train_x_tensor.shape[0])
            train_batch_count = int(ceil(n_train / batch_size))
            batch_ready_mark = time.perf_counter()
            permutation = torch.randperm(n_train, device=device)
            for start_idx in _split_batch_starts(n_train, batch_size):
                wait_start = batch_ready_mark
                batch_indices = permutation[start_idx : start_idx + batch_size]
                xb = train_x_tensor.index_select(0, batch_indices)
                yb = train_y_tensor.index_select(0, batch_indices)
                maybe_cuda_synchronize(device if profiler.enabled else None)
                batch_wait_sec = time.perf_counter() - wait_start

                h2d_sec = 0.0
                optimizer.zero_grad(set_to_none=True)

                forward_start = time.perf_counter()
                pred = model(xb)
                loss = criterion(pred, yb)
                maybe_cuda_synchronize(device if profiler.enabled else None)
                forward_sec = time.perf_counter() - forward_start

                backward_start = time.perf_counter()
                loss.backward()
                maybe_cuda_synchronize(device if profiler.enabled else None)
                backward_sec = time.perf_counter() - backward_start

                optimizer_start = time.perf_counter()
                optimizer.step()
                maybe_cuda_synchronize(device if profiler.enabled else None)
                optimizer_sec = time.perf_counter() - optimizer_start

                metrics_start = time.perf_counter()
                train_loss_total += loss.detach()
                train_mae_total += torch.mean(torch.abs(pred.detach() - yb))
                metrics_sec = time.perf_counter() - metrics_start

                if profiler.enabled:
                    profiler.batch_timings.add_sample(
                        batch_wait_sec=batch_wait_sec,
                        h2d_sec=h2d_sec,
                        forward_sec=forward_sec,
                        backward_sec=backward_sec,
                        optimizer_sec=optimizer_sec,
                        metrics_sec=metrics_sec,
                    )
                batch_ready_mark = time.perf_counter()
        else:
            train_iter = iter(train_loader)
            batch_ready_mark = time.perf_counter()
            while True:
                wait_start = batch_ready_mark
                try:
                    xb, yb = next(train_iter)
                except StopIteration:
                    break
                batch_wait_sec = time.perf_counter() - wait_start

                h2d_start = time.perf_counter()
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                maybe_cuda_synchronize(device if profiler.enabled else None)
                h2d_sec = time.perf_counter() - h2d_start

                optimizer.zero_grad(set_to_none=True)

                forward_start = time.perf_counter()
                pred = model(xb)
                loss = criterion(pred, yb)
                maybe_cuda_synchronize(device if profiler.enabled else None)
                forward_sec = time.perf_counter() - forward_start

                backward_start = time.perf_counter()
                loss.backward()
                maybe_cuda_synchronize(device if profiler.enabled else None)
                backward_sec = time.perf_counter() - backward_start

                optimizer_start = time.perf_counter()
                optimizer.step()
                maybe_cuda_synchronize(device if profiler.enabled else None)
                optimizer_sec = time.perf_counter() - optimizer_start

                metrics_start = time.perf_counter()
                train_loss_total += loss.detach()
                train_mae_total += torch.mean(torch.abs(pred.detach() - yb))
                train_batch_count += 1
                metrics_sec = time.perf_counter() - metrics_start

                if profiler.enabled:
                    profiler.batch_timings.add_sample(
                        batch_wait_sec=batch_wait_sec,
                        h2d_sec=h2d_sec,
                        forward_sec=forward_sec,
                        backward_sec=backward_sec,
                        optimizer_sec=optimizer_sec,
                        metrics_sec=metrics_sec,
                    )
                batch_ready_mark = time.perf_counter()
        profiler.train_loop_total_sec += time.perf_counter() - train_loop_start

        model.eval()
        val_loss_total = torch.zeros((), device=device)
        val_mae_total = torch.zeros((), device=device)
        val_batch_count = 0
        validation_loop_start = time.perf_counter()
        with torch.no_grad():
            if using_device_resident_batches:
                n_val = int(val_x_tensor.shape[0])
                val_batch_count = int(ceil(n_val / batch_size))
                for start_idx in _split_batch_starts(n_val, batch_size):
                    xb = val_x_tensor[start_idx : start_idx + batch_size]
                    yb = val_y_tensor[start_idx : start_idx + batch_size]
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    val_loss_total += loss.detach()
                    val_mae_total += torch.mean(torch.abs(pred.detach() - yb))
            else:
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    val_loss_total += loss.detach()
                    val_mae_total += torch.mean(torch.abs(pred.detach() - yb))
                    val_batch_count += 1
        profiler.validation_loop_total_sec += time.perf_counter() - validation_loop_start

        train_loss = _mean_batch_metric(train_loss_total, train_batch_count)
        val_loss = _mean_batch_metric(val_loss_total, val_batch_count)
        train_mae = _mean_batch_metric(train_mae_total, train_batch_count)
        val_mae = _mean_batch_metric(val_mae_total, val_batch_count)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        elapsed = time.time() - global_start
        epoch_duration = time.time() - epoch_start
        profiler.record_epoch_duration(epoch_duration)
        remaining = max(epochs - epoch, 0) * epoch_duration
        if logger is not None:
            logger.info(
                "[epoch %03d/%03d] train_loss=%.6f val_loss=%.6f train_mae=%.6f val_mae=%.6f elapsed=%.2fm eta=%.2fm",
                epoch,
                epochs,
                train_loss,
                val_loss,
                train_mae,
                val_mae,
                elapsed / 60.0,
                remaining / 60.0,
            )

        if epochs_without_improvement >= patience:
            if logger is not None:
                logger.info("Early stopping triggered after %s epochs", epoch)
            break

    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(
        model=model,
        history=history,
        test_x=test_x,
        model_config=model_config,
        loader_prep_sec=loader_prep_sec,
        batching_strategy="device_resident" if using_device_resident_batches else "dataloader",
        resident_data_bytes=resident_data_bytes,
    )


def predict_model(model: nn.Module, test_x: torch.Tensor, y_scaler, device: torch.device):
    model.eval()
    with torch.no_grad():
        pred_scaled = model(test_x.to(device)).detach().cpu().numpy()
    pred = y_scaler.inverse_transform(pred_scaled).ravel()
    return pred
