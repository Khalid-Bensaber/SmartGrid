from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from smartgrid.common.utils import set_seed
from smartgrid.models.mlp import TorchMLP


@dataclass(slots=True)
class TrainResult:
    model: nn.Module
    history: dict
    test_x: torch.Tensor
    model_config: dict


def make_loaders(X_train, y_train, X_val, y_val, X_test, batch_size: int, num_workers: int):
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
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_x


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
) -> TrainResult:
    set_seed(seed)

    train_loader, val_loader, test_x = make_loaders(
        X_train, y_train, X_val, y_val, X_test,
        batch_size=batch_size,
        num_workers=num_workers,
    )

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
        train_losses = []
        train_maes = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_maes.append(torch.mean(torch.abs(pred - yb)).item())

        model.eval()
        val_losses = []
        val_maes = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
                val_maes.append(torch.mean(torch.abs(pred - yb)).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        train_mae = float(np.mean(train_maes))
        val_mae = float(np.mean(val_maes))

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
    return TrainResult(model=model, history=history, test_x=test_x, model_config=model_config)


def predict_model(model: nn.Module, test_x: torch.Tensor, y_scaler, device: torch.device):
    model.eval()
    with torch.no_grad():
        pred_scaled = model(test_x.to(device)).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_scaled).ravel()
    return pred
