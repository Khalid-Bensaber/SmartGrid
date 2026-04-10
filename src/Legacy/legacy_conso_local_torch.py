#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch version of the legacy local consumption runner.

What this file keeps from the current legacy_conso_local.py:
- same general CLI spirit
- same clean CSV inputs
- same rebuilt total target from 4 building columns
- same manual lag features (J-7, J-1 ... J-6)
- same chronological split logic
- same notebook-aligned metrics
- same comparison against the old legacy forecast CSV
- same export style for notebook analysis

What changes:
- training/inference backend is PyTorch instead of TensorFlow / sklearn
- model saving/loading uses torch checkpoints
- training loop is explicit and commented for easier study

This file is intentionally monolithic so you we study it quickly, run it quickly,
and then split it later when we put the GitHub structure in place.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
import warnings
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
TOTAL_COLUMNS = ["Ptot_HA", "Ptot_HEI_13RT", "Ptot_HEI_5RNS", "Ptot_RIZOMM"]
OLD_FORECAST_COLUMNS = [
    "Ptot_HA_Forecast",
    "Ptot_HEI_13RT_Forecast",
    "Ptot_HEI_5RNS_Forecast",
    "Ptot_RIZOMM_Forecast",
]
FORECAST_SCHEMA_COLUMNS = [
    "name",
    "Date",
    "Ptot_HA_Forecast",
    "Ptot_HEI_13RT_Forecast",
    "Ptot_HEI_5RNS_Forecast",
    "Ptot_Ilot_Forecast",
    "Ptot_RIZOMM_Forecast",
]

DEFAULT_TARGET_NAME = "tot"
DEFAULT_AIRTEMP_VALUE = 15.0
N_STEPS_PER_DAY = 144
N_LAG_DAYS = 7
LAG_POINTS = N_STEPS_PER_DAY * N_LAG_DAYS
TOL_REL = 0.01
TOL_ABS = 5000.0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PyTorch local consumption trainer/backtester")
    p.add_argument(
        "--historical-csv",
        type=str,
        default="data/processed/2025-12-02_10_15_previsions_data_conso_historiques_clean.csv",
    )
    p.add_argument(
        "--holidays-xlsx",
        type=str,
        default="data/processed/Holidays.xlsx",
    )
    p.add_argument(
        "--benchmark-csv",
        type=str,
        default="data/processed/conso/2025-12-02_10_15_previsions_data_conso_prev_ptot_clean.csv",
    )
    p.add_argument("--output-dir", type=str, default="data/processed/conso")
    p.add_argument("--output-filename", type=str, default="legacy_local_prev_ptot_clean.csv")
    p.add_argument("--date-col", type=str, default="Date")

    # Split controls
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--train-end-date", type=str, default=None, help="Optional explicit train end date YYYY-MM-DD")
    p.add_argument("--val-end-date", type=str, default=None, help="Optional explicit val end date YYYY-MM-DD")

    # Model / training controls
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=720)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--hidden-layers", type=str, default="1024,512")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resume-model", type=str, default=None)
    p.add_argument("--skip-train", action="store_true", help="Load a saved PyTorch model + scalers without retraining")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # Analysis / export controls
    p.add_argument("--analysis-date", type=str, default=None, help="Day to export/analyse (YYYY-MM-DD)")
    p.add_argument("--analysis-days", type=int, default=1, help="Number of consecutive days to export from analysis-date")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_hidden_layers(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def get_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info(device: torch.device) -> dict:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": None,
    }
    if device.type == "cuda":
        try:
            info["gpu_name"] = torch.cuda.get_device_name(device)
        except Exception:
            pass
    return info


def log_block(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# -----------------------------------------------------------------------------
# Data loading / feature engineering
# -----------------------------------------------------------------------------
def load_holiday_sets(holidays_xlsx: str) -> tuple[set, set]:
    """Read the Excel file used in the legacy pipeline and extract holiday sets.

    We intentionally keep the same rough convention as the legacy script:
    - column 'Unnamed: 0' -> holidays
    - column 'Unnamed: 2' -> special days
    """
    xls = pd.ExcelFile(holidays_xlsx)
    holiday_dates = set()
    special_dates = set()

    for sheet in xls.sheet_names:
        df = pd.read_excel(holidays_xlsx, sheet_name=sheet)
        if "Unnamed: 0" in df.columns:
            s = pd.to_datetime(df["Unnamed: 0"], errors="coerce").dropna().dt.date
            holiday_dates.update(s.tolist())
        if "Unnamed: 2" in df.columns:
            s2 = pd.to_datetime(df["Unnamed: 2"], errors="coerce").dropna().dt.date
            special_dates.update(s2.tolist())

    return holiday_dates, special_dates


def load_history(csv_path: str, date_col: str = "Date") -> pd.DataFrame:
    """Load the clean historical CSV and rebuild the total target."""
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    missing_cols = [c for c in TOTAL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required history columns: {missing_cols}")

    # Total used by the notebook and by our local tests.
    df[DEFAULT_TARGET_NAME] = df[TOTAL_COLUMNS].sum(axis=1)

    # Keep the clean file as-is. We only add a default temperature if missing.
    if "Airtemp" not in df.columns:
        if "AirTemp" in df.columns:
            df["Airtemp"] = df["AirTemp"]
        else:
            df["Airtemp"] = DEFAULT_AIRTEMP_VALUE

    return df


def add_calendar_features(df: pd.DataFrame, holiday_dates: set, special_dates: set, date_col: str) -> pd.DataFrame:
    """Add the exact style of calendar features used in the legacy logic."""
    out = df.copy()
    dt = out[date_col]

    out["day_of_year"] = dt.dt.dayofyear
    out["minute_of_day"] = dt.dt.hour * 60 + dt.dt.minute
    out["weekday"] = dt.dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["month"] = dt.dt.month
    out["date_only"] = dt.dt.date
    out["is_holiday"] = out["date_only"].isin(holiday_dates).astype(int)
    out["is_special_day"] = out["date_only"].isin(special_dates).astype(int)

    # Same convention as before: if the day is special, it overrides holiday.
    out.loc[out["is_special_day"] == 1, "is_holiday"] = 0
    return out


def add_manual_lag_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Add the legacy manual lag features.

    Important: this is NOT a sequence model. We give the MLP a fixed flat vector that
    explicitly contains past values at selected day lags.
    """
    out = df.copy()
    out["lag_d7"] = out[target_col].shift(144 * 7)
    out["lag_d1"] = out[target_col].shift(144 * 1)
    out["lag_d2"] = out[target_col].shift(144 * 2)
    out["lag_d3"] = out[target_col].shift(144 * 3)
    out["lag_d4"] = out[target_col].shift(144 * 4)
    out["lag_d5"] = out[target_col].shift(144 * 5)
    out["lag_d6"] = out[target_col].shift(144 * 6)
    return out


def build_feature_table(hist_df: pd.DataFrame, holiday_dates: set, special_dates: set, date_col: str = "Date"):
    """Create the full feature table used by the PyTorch MLP."""
    df = hist_df.copy()
    df = add_calendar_features(df, holiday_dates, special_dates, date_col)
    df = add_manual_lag_features(df, DEFAULT_TARGET_NAME)

    feature_cols = [
        "day_of_year",
        "minute_of_day",
        "weekday",
        "is_weekend",
        "month",
        "is_holiday",
        "is_special_day",
        "Airtemp",
        "lag_d7",
        "lag_d1",
        "lag_d2",
        "lag_d3",
        "lag_d4",
        "lag_d5",
        "lag_d6",
    ]

    # We drop only rows that are unusable for these features / target.
    df = df.dropna(subset=feature_cols + [DEFAULT_TARGET_NAME]).reset_index(drop=True)
    return df, feature_cols


# -----------------------------------------------------------------------------
# Splits
# -----------------------------------------------------------------------------
def chronological_split_by_ratio(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    n = len(df)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:i_train].copy()
    val_df = df.iloc[i_train:i_val].copy()
    test_df = df.iloc[i_val:].copy()
    return train_df, val_df, test_df


def chronological_split_by_dates(df: pd.DataFrame, date_col: str, train_end_date: str, val_end_date: str):
    train_end = pd.to_datetime(train_end_date)
    val_end = pd.to_datetime(val_end_date)

    train_df = df[df[date_col] <= train_end].copy()
    val_df = df[(df[date_col] > train_end) & (df[date_col] <= val_end)].copy()
    test_df = df[df[date_col] > val_end].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Date-based split produced an empty split. Check --train-end-date / --val-end-date.")
    return train_df, val_df, test_df


def make_splits(df: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.train_end_date and args.val_end_date:
        return chronological_split_by_dates(df, args.date_col, args.train_end_date, args.val_end_date)
    return chronological_split_by_ratio(df, train_ratio=args.train_ratio, val_ratio=args.val_ratio)


# -----------------------------------------------------------------------------
# PyTorch model
# -----------------------------------------------------------------------------
class TorchMLP(nn.Module):
    """Simple configurable dense MLP for tabular regression."""

    def __init__(self, input_dim: int, hidden_layers: Iterable[int], dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for units in hidden_layers:
            layers.append(nn.Linear(prev, units))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = units
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


def save_model_artifacts(model: nn.Module, x_scaler, y_scaler, output_dir: Path, model_config: dict):
    model_path = output_dir / "legacy_model_torch.pt"
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": model_config,
    }
    torch.save(checkpoint, model_path)

    x_scaler_path = output_dir / "x_scaler.save"
    y_scaler_path = output_dir / "y_scaler.save"
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)
    return model_path, x_scaler_path, y_scaler_path


def load_saved_artifacts(output_dir: Path, device: torch.device):
    model_path = output_dir / "legacy_model_torch.pt"
    x_scaler_path = output_dir / "x_scaler.save"
    y_scaler_path = output_dir / "y_scaler.save"

    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint["model_config"]
    model = TorchMLP(
        input_dim=model_config["input_dim"],
        hidden_layers=model_config["hidden_layers"],
        dropout=model_config.get("dropout", 0.0),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    return model, x_scaler, y_scaler, model_path, model_config


def train_model_pytorch(
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
    resume_model: str | None = None,
):
    """Explicit PyTorch training loop with validation, early stopping and ETA logs."""
    set_seed(seed)

    train_loader, val_loader, test_x = make_loaders(
        X_train, y_train, X_val, y_val, X_test,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if resume_model:
        checkpoint = torch.load(resume_model, map_location=device)
        model_config = checkpoint["model_config"]
        model = TorchMLP(
            input_dim=model_config["input_dim"],
            hidden_layers=model_config["hidden_layers"],
            dropout=model_config.get("dropout", 0.0),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"[INFO] resumed PyTorch model from: {resume_model}")
    else:
        model = TorchMLP(input_dim=X_train.shape[1], hidden_layers=hidden_layers, dropout=dropout).to(device)

    print("\n[MODEL] PyTorch model:")
    print(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    epochs_without_improvement = 0
    global_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ---------------------------
        # Training step
        # ---------------------------
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

        # ---------------------------
        # Validation step
        # ---------------------------
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

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - global_start
        avg_epoch_time = elapsed / epoch
        remaining = (epochs - epoch) * avg_epoch_time

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"mae={train_mae:.6f} val_mae={val_mae:.6f} "
            f"time={epoch_time:.1f}s eta~={remaining/60:.1f} min"
        )

        # Early stopping on validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"[INFO] early stopping triggered after epoch {epoch} (patience={patience})")
            break

    # Restore best weights before prediction/export.
    model.load_state_dict(best_state)
    model.eval()

    model_config = {
        "input_dim": X_train.shape[1],
        "hidden_layers": list(hidden_layers),
        "dropout": dropout,
    }
    return model, history, test_x, model_config


def predict_model_pytorch(model: nn.Module, test_x: torch.Tensor, y_scaler, device: torch.device):
    model.eval()
    with torch.no_grad():
        pred_scaled = model(test_x.to(device)).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_scaled).ravel()
    return pred


# -----------------------------------------------------------------------------
# Metrics (aligned with notebook)
# -----------------------------------------------------------------------------
def build_metrics_df(merged: pd.DataFrame, real_col: str, fc_col: str, tol_abs: float, tol_rel: float):
    m = merged.copy()
    m = m.rename(columns={real_col: "Ptot_TOTAL_REAL", fc_col: "Ptot_TOTAL_FC"})
    m["error"] = m["Ptot_TOTAL_FC"] - m["Ptot_TOTAL_REAL"]
    m["abs_error"] = m["error"].abs()
    m["tol_point"] = np.maximum(tol_abs, tol_rel * m["Ptot_TOTAL_REAL"].abs())
    m["in_tol"] = m["abs_error"] <= m["tol_point"]
    denom = m["Ptot_TOTAL_REAL"].abs().clip(lower=tol_abs)
    m["ape_%"] = 100.0 * (m["abs_error"] / denom)
    return m


def compute_metrics_v2(g: pd.DataFrame) -> dict:
    err = g["error"]
    abs_err = g["abs_error"]
    real = g["Ptot_TOTAL_REAL"]
    fc = g["Ptot_TOTAL_FC"]

    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err**2).mean()))
    bias = float(err.mean())
    mape = float(g["ape_%"].mean())

    under = err[err < 0]
    over = err[err > 0]
    under_mae = float(under.abs().mean()) if len(under) else None
    over_mae = float(over.abs().mean()) if len(over) else None

    p95 = float(abs_err.quantile(0.95))
    p99 = float(abs_err.quantile(0.99))
    in_tol = float(100.0 * g["in_tol"].mean())

    denom = (real.abs() + fc.abs()).clip(lower=TOL_ABS)
    smape = float((200.0 * abs_err / denom).mean())

    corr_raw = abs_err.corr(real)
    corr = float(corr_raw) if pd.notna(corr_raw) else None

    dr = real.diff()
    dfc = fc.diff()
    derr = dfc - dr
    ramp_rmse = float(np.sqrt((derr**2).mean())) if len(derr.dropna()) > 0 else None

    return {
        "count": int(len(g)),
        "MAE": mae,
        "RMSE": rmse,
        "Bias(ME)": bias,
        "MAPE%": mape,
        "InTolerance%": in_tol,
        "P95AbsError": p95,
        "P99AbsError": p99,
        "UnderShare%": float(100.0 * (err < 0).mean()),
        "OverShare%": float(100.0 * (err > 0).mean()),
        "Under_MAE": under_mae,
        "Over_MAE": over_mae,
        "SMAPE%": smape,
        "CorrAbsErr_vs_Real": corr,
        "RampingError_RMSE": ramp_rmse,
    }


def seasonal_naive_weekly(real: pd.Series, lag="7D") -> pd.Series:
    lag = pd.Timedelta(lag)
    values = real.reindex(real.index - lag).to_numpy()
    return pd.Series(values, index=real.index)


def compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mbe = float(np.mean(y_pred - y_true))
    denom = np.clip(np.abs(y_true), TOL_ABS, None)
    mape = float(np.mean(np.abs(y_pred - y_true) / denom) * 100.0)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MBE": mbe,
        "MAPE%": mape,
    }


# -----------------------------------------------------------------------------
# Benchmark / export helpers
# -----------------------------------------------------------------------------
def load_old_benchmark(benchmark_csv: str, date_col: str = "Date") -> pd.DataFrame | None:
    if benchmark_csv is None:
        return None
    p = Path(benchmark_csv)
    if not p.exists():
        warnings.warn(f"Benchmark CSV not found: {benchmark_csv}")
        return None

    old = pd.read_csv(p)
    old[date_col] = pd.to_datetime(old[date_col], errors="coerce")
    old = old.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    missing = [c for c in OLD_FORECAST_COLUMNS if c not in old.columns]
    if missing:
        warnings.warn(f"Benchmark CSV missing columns needed for notebook total: {missing}")
        return None

    old["OldLegacy_TOTAL_Forecast"] = old[OLD_FORECAST_COLUMNS].sum(axis=1)
    keep_cols = [date_col, "OldLegacy_TOTAL_Forecast"]
    if "Ptot_Ilot_Forecast" in old.columns:
        keep_cols.append("Ptot_Ilot_Forecast")
    return old[keep_cols].copy()


def pick_analysis_day(backtest: pd.DataFrame, benchmark: pd.DataFrame | None, date_col: str, requested_day: str | None):
    all_days = sorted(backtest[date_col].dt.date.unique())
    if benchmark is not None:
        bench_days = set(benchmark[date_col].dt.date.unique())
        intersection_days = [d for d in all_days if d in bench_days]
        if intersection_days:
            all_days = intersection_days
    if not all_days:
        raise RuntimeError("No valid day found in backtest / benchmark intersection.")

    if requested_day is not None:
        day = pd.to_datetime(requested_day).date()
        if day not in all_days:
            raise RuntimeError(f"Requested analysis day {day} is not available in valid test days.")
        return day
    return all_days[0]


def make_notebook_export_legacy_schema(df_day: pd.DataFrame, date_col: str) -> pd.DataFrame:
    export = pd.DataFrame(
        {
            "name": "CONSO_Prevision_Data",
            "Date": df_day[date_col],
            "Ptot_HA_Forecast": np.nan,
            "Ptot_HEI_13RT_Forecast": np.nan,
            "Ptot_HEI_5RNS_Forecast": np.nan,
            "Ptot_Ilot_Forecast": df_day["Ptot_TOTAL_Forecast"].values,
            "Ptot_RIZOMM_Forecast": np.nan,
        }
    )
    return export[FORECAST_SCHEMA_COLUMNS]


def make_total_export(df_day: pd.DataFrame, date_col: str) -> pd.DataFrame:
    cols = [date_col, DEFAULT_TARGET_NAME, "Ptot_TOTAL_Forecast"]
    if "OldLegacy_TOTAL_Forecast" in df_day.columns:
        cols.append("OldLegacy_TOTAL_Forecast")
    out = df_day[cols].copy()
    out = out.rename(columns={DEFAULT_TARGET_NAME: "Ptot_TOTAL_Real"})
    return out


def compute_new_vs_old_comparison(metrics_new: dict | None, metrics_old: dict | None) -> dict | None:
    if metrics_new is None or metrics_old is None:
        return None

    def pct_skill(new_val, old_val):
        if new_val is None or old_val in (None, 0):
            return None
        return 100.0 * (1.0 - new_val / old_val)

    def delta(new_val, old_val):
        if new_val is None or old_val is None:
            return None
        return new_val - old_val

    return {
        "MAE_new": metrics_new.get("MAE"),
        "MAE_old": metrics_old.get("MAE"),
        "MAE_delta_new_minus_old": delta(metrics_new.get("MAE"), metrics_old.get("MAE")),
        "MAE_skill%_new_vs_old": pct_skill(metrics_new.get("MAE"), metrics_old.get("MAE")),
        "RMSE_new": metrics_new.get("RMSE"),
        "RMSE_old": metrics_old.get("RMSE"),
        "RMSE_delta_new_minus_old": delta(metrics_new.get("RMSE"), metrics_old.get("RMSE")),
        "RMSE_skill%_new_vs_old": pct_skill(metrics_new.get("RMSE"), metrics_old.get("RMSE")),
        "Bias_new": metrics_new.get("Bias(ME)"),
        "Bias_old": metrics_old.get("Bias(ME)"),
        "Bias_delta_new_minus_old": delta(metrics_new.get("Bias(ME)"), metrics_old.get("Bias(ME)")),
        "InTolerance%_new": metrics_new.get("InTolerance%"),
        "InTolerance%_old": metrics_old.get("InTolerance%"),
        "InTolerance%_delta_new_minus_old": delta(metrics_new.get("InTolerance%"), metrics_old.get("InTolerance%")),
        "P95AbsError_new": metrics_new.get("P95AbsError"),
        "P95AbsError_old": metrics_old.get("P95AbsError"),
        "P95AbsError_delta_new_minus_old": delta(metrics_new.get("P95AbsError"), metrics_old.get("P95AbsError")),
        "P99AbsError_new": metrics_new.get("P99AbsError"),
        "P99AbsError_old": metrics_old.get("P99AbsError"),
        "P99AbsError_delta_new_minus_old": delta(metrics_new.get("P99AbsError"), metrics_old.get("P99AbsError")),
        "MAPE%_new": metrics_new.get("MAPE%"),
        "MAPE%_old": metrics_old.get("MAPE%"),
        "MAPE%_delta_new_minus_old": delta(metrics_new.get("MAPE%"), metrics_old.get("MAPE%")),
        "SMAPE%_new": metrics_new.get("SMAPE%"),
        "SMAPE%_old": metrics_old.get("SMAPE%"),
        "SMAPE%_delta_new_minus_old": delta(metrics_new.get("SMAPE%"), metrics_old.get("SMAPE%")),
        "RampingError_RMSE_new": metrics_new.get("RampingError_RMSE"),
        "RampingError_RMSE_old": metrics_old.get("RampingError_RMSE"),
        "RampingError_RMSE_delta_new_minus_old": delta(metrics_new.get("RampingError_RMSE"), metrics_old.get("RampingError_RMSE")),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    device = get_device(args.device)
    device_info = get_device_info(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_block("ENVIRONMENT")
    print(f"[INFO] torch version:             {device_info['torch_version']}")
    print(f"[INFO] cuda available:           {device_info['cuda_available']}")
    print(f"[INFO] selected device:          {device_info['device']}")
    if device_info["gpu_name"]:
        print(f"[INFO] gpu name:                 {device_info['gpu_name']}")

    log_block("LOADING DATA")
    holiday_dates, special_dates = load_holiday_sets(args.holidays_xlsx)
    hist = load_history(args.historical_csv, date_col=args.date_col)
    feat_df, feature_cols = build_feature_table(hist, holiday_dates, special_dates, date_col=args.date_col)
    train_df, val_df, test_df = make_splits(feat_df, args)

    print(f"[INFO] total history rows raw:    {len(hist)}")
    print(f"[INFO] usable rows after lags:    {len(feat_df)}")
    print(f"[INFO] train rows:                {len(train_df)}")
    print(f"[INFO] val rows:                  {len(val_df)}")
    print(f"[INFO] test rows:                 {len(test_df)}")
    print(f"[INFO] train range:               {train_df[args.date_col].min()} -> {train_df[args.date_col].max()}")
    print(f"[INFO] val range:                 {val_df[args.date_col].min()} -> {val_df[args.date_col].max()}")
    print(f"[INFO] test range:                {test_df[args.date_col].min()} -> {test_df[args.date_col].max()}")
    print(f"[INFO] feature columns ({len(feature_cols)}): {feature_cols}")

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[[DEFAULT_TARGET_NAME]].to_numpy(dtype=float)
    X_val = val_df[feature_cols].to_numpy(dtype=float)
    y_val = val_df[[DEFAULT_TARGET_NAME]].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[DEFAULT_TARGET_NAME].to_numpy(dtype=float)

    # Same scaling strategy as the TensorFlow version.
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    X_val_scaled = x_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val)
    X_test_scaled = x_scaler.transform(X_test)

    log_block("TRAINING / LOADING MODEL")
    if args.skip_train:
        print("[INFO] skip-train enabled -> loading saved model + scalers from output dir")
        model, x_scaler, y_scaler, model_path, model_config = load_saved_artifacts(output_dir, device)
        history = {"loaded_from_disk": True}
        test_x = torch.tensor(X_test_scaled, dtype=torch.float32)
    else:
        train_start = time.time()
        model, history, test_x, model_config = train_model_pytorch(
            X_train=X_train_scaled,
            y_train=y_train_scaled,
            X_val=X_val_scaled,
            y_val=y_val_scaled,
            X_test=X_test_scaled,
            hidden_layers=hidden_layers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            dropout=args.dropout,
            seed=args.seed,
            num_workers=args.num_workers,
            device=device,
            resume_model=args.resume_model,
        )
        train_duration_sec = time.time() - train_start
        print(f"[INFO] training duration: {train_duration_sec/60:.2f} min")
        model_path, _, _ = save_model_artifacts(model, x_scaler, y_scaler, output_dir, model_config)

    log_block("BACKTEST")
    y_pred = predict_model_pytorch(model, test_x, y_scaler, device=device)
    basic_metrics = compute_basic_metrics(y_test, y_pred)
    for k, v in basic_metrics.items():
        print(f"[INFO] basic {k}: {v}")

    backtest = test_df[[args.date_col, DEFAULT_TARGET_NAME, "lag_d1"]].copy()
    backtest["Ptot_TOTAL_Forecast"] = y_pred
    backtest["name"] = "CONSO_Prevision_Data"

    benchmark = load_old_benchmark(args.benchmark_csv, date_col=args.date_col)
    if benchmark is not None:
        backtest = backtest.merge(benchmark, on=args.date_col, how="left")

    # Notebook-aligned metrics for new model.
    model_eval = build_metrics_df(
        merged=backtest[[args.date_col, DEFAULT_TARGET_NAME, "Ptot_TOTAL_Forecast"]].set_index(args.date_col),
        real_col=DEFAULT_TARGET_NAME,
        fc_col="Ptot_TOTAL_Forecast",
        tol_abs=TOL_ABS,
        tol_rel=TOL_REL,
    )

    # Weekly seasonal naive used for quick baseline comparison.
    naive_df = model_eval.copy()
    naive_df["Ptot_TOTAL_FC"] = seasonal_naive_weekly(model_eval["Ptot_TOTAL_REAL"], lag="7D")
    naive_eval = build_metrics_df(
        merged=naive_df.reset_index(),
        real_col="Ptot_TOTAL_REAL",
        fc_col="Ptot_TOTAL_FC",
        tol_abs=TOL_ABS,
        tol_rel=TOL_REL,
    ).set_index(args.date_col if args.date_col in naive_df.reset_index().columns else model_eval.index.name)

    idx_valid = naive_eval["Ptot_TOTAL_FC"].notna()
    model_eval_aligned = model_eval.loc[idx_valid]
    naive_eval_aligned = naive_eval.loc[idx_valid]

    metrics_model = compute_metrics_v2(model_eval_aligned)
    metrics_naive_weekly = compute_metrics_v2(naive_eval_aligned)
    metrics_old_legacy = None
    metrics_model_on_old_overlap = None
    comparison_new_vs_old = None
    old_overlap_count = 0

    if "OldLegacy_TOTAL_Forecast" in backtest.columns:
        overlap = backtest.dropna(subset=["OldLegacy_TOTAL_Forecast"]).copy()
        old_overlap_count = int(len(overlap))
        if len(overlap) > 0:
            old_eval = build_metrics_df(
                merged=overlap[[args.date_col, DEFAULT_TARGET_NAME, "OldLegacy_TOTAL_Forecast"]].set_index(args.date_col),
                real_col=DEFAULT_TARGET_NAME,
                fc_col="OldLegacy_TOTAL_Forecast",
                tol_abs=TOL_ABS,
                tol_rel=TOL_REL,
            )
            model_overlap_eval = build_metrics_df(
                merged=overlap[[args.date_col, DEFAULT_TARGET_NAME, "Ptot_TOTAL_Forecast"]].set_index(args.date_col),
                real_col=DEFAULT_TARGET_NAME,
                fc_col="Ptot_TOTAL_Forecast",
                tol_abs=TOL_ABS,
                tol_rel=TOL_REL,
            )
            metrics_old_legacy = compute_metrics_v2(old_eval)
            metrics_model_on_old_overlap = compute_metrics_v2(model_overlap_eval)
            comparison_new_vs_old = compute_new_vs_old_comparison(metrics_model_on_old_overlap, metrics_old_legacy)

    comparison = {
        "MASE (MAE_model / MAE_naive_weekly)": (
            metrics_model["MAE"] / metrics_naive_weekly["MAE"] if metrics_naive_weekly["MAE"] else None
        ),
        "MAE_skill% vs naive_weekly": (
            100.0 * (1.0 - metrics_model["MAE"] / metrics_naive_weekly["MAE"])
            if metrics_naive_weekly["MAE"] else None
        ),
        "RMSE_skill% vs naive_weekly": (
            100.0 * (1.0 - metrics_model["RMSE"] / metrics_naive_weekly["RMSE"])
            if metrics_naive_weekly["RMSE"] else None
        ),
    }

    log_block("SELECTING ANALYSIS DAY")
    analysis_day = pick_analysis_day(backtest, benchmark, args.date_col, args.analysis_date)
    start_day = pd.Timestamp(analysis_day)
    end_day = start_day + pd.Timedelta(days=args.analysis_days)
    mask = (backtest[args.date_col] >= start_day) & (backtest[args.date_col] < end_day)
    day_df = backtest.loc[mask].copy()

    print(f"[INFO] selected analysis day:     {analysis_day}")
    print(f"[INFO] analysis days exported:    {args.analysis_days}")
    print(f"[INFO] selected rows:             {len(day_df)}")

    notebook_export_day = make_notebook_export_legacy_schema(day_df, args.date_col)
    notebook_export_path = output_dir / args.output_filename
    notebook_export_day.to_csv(notebook_export_path, index=False)

    total_export_day = make_total_export(day_df, args.date_col)
    total_export_path = output_dir / f"total_{Path(args.output_filename).stem}.csv"
    total_export_day.to_csv(total_export_path, index=False)

    backtest_path = output_dir / "backtest_predictions.csv"
    backtest.to_csv(backtest_path, index=False)
    selected_day_path = output_dir / f"selected_day_{analysis_day}.csv"
    day_df.to_csv(selected_day_path, index=False)

    day_summary = {
        "analysis_day": str(analysis_day),
        "n_points": int(len(day_df)),
        "date_min": str(day_df[args.date_col].min()) if len(day_df) else None,
        "date_max": str(day_df[args.date_col].max()) if len(day_df) else None,
    }
    if len(day_df):
        model_day_eval = build_metrics_df(
            merged=day_df[[args.date_col, DEFAULT_TARGET_NAME, "Ptot_TOTAL_Forecast"]].set_index(args.date_col),
            real_col=DEFAULT_TARGET_NAME,
            fc_col="Ptot_TOTAL_Forecast",
            tol_abs=TOL_ABS,
            tol_rel=TOL_REL,
        )
        day_summary["metrics_model_day"] = compute_metrics_v2(model_day_eval)

        if "OldLegacy_TOTAL_Forecast" in day_df.columns and day_df["OldLegacy_TOTAL_Forecast"].notna().any():
            old_day = day_df.dropna(subset=["OldLegacy_TOTAL_Forecast"])
            old_day_eval = build_metrics_df(
                merged=old_day[[args.date_col, DEFAULT_TARGET_NAME, "OldLegacy_TOTAL_Forecast"]].set_index(args.date_col),
                real_col=DEFAULT_TARGET_NAME,
                fc_col="OldLegacy_TOTAL_Forecast",
                tol_abs=TOL_ABS,
                tol_rel=TOL_REL,
            )
            model_day_overlap_eval = build_metrics_df(
                merged=old_day[[args.date_col, DEFAULT_TARGET_NAME, "Ptot_TOTAL_Forecast"]].set_index(args.date_col),
                real_col=DEFAULT_TARGET_NAME,
                fc_col="Ptot_TOTAL_Forecast",
                tol_abs=TOL_ABS,
                tol_rel=TOL_REL,
            )
            day_summary["metrics_old_legacy_day"] = compute_metrics_v2(old_day_eval)
            day_summary["metrics_model_day_on_old_overlap"] = compute_metrics_v2(model_day_overlap_eval)
            day_summary["comparison_new_vs_old_day"] = compute_new_vs_old_comparison(
                day_summary["metrics_model_day_on_old_overlap"],
                day_summary["metrics_old_legacy_day"],
            )

    summary = {
        "backend": "pytorch",
        "device_info": device_info,
        "feature_columns": feature_cols,
        "hidden_layers": list(hidden_layers),
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_end_date": args.train_end_date,
        "val_end_date": args.val_end_date,
        "metrics_basic": basic_metrics,
        "metrics_model": metrics_model,
        "metrics_naive_weekly": metrics_naive_weekly,
        "metrics_old_legacy": metrics_old_legacy,
        "metrics_model_on_old_overlap": metrics_model_on_old_overlap,
        "old_overlap_count": old_overlap_count,
        "comparison": comparison,
        "comparison_new_vs_old": comparison_new_vs_old,
        "n_total_rows": int(len(feat_df)),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "train_date_min": str(train_df[args.date_col].min()),
        "train_date_max": str(train_df[args.date_col].max()),
        "val_date_min": str(val_df[args.date_col].min()),
        "val_date_max": str(val_df[args.date_col].max()),
        "test_date_min": str(test_df[args.date_col].min()),
        "test_date_max": str(test_df[args.date_col].max()),
        "selected_analysis_day": str(analysis_day),
        "analysis_days": args.analysis_days,
        "model_path": str(Path(model_path).resolve()),
        "output_forecast_csv": str(notebook_export_path.resolve()),
        "output_total_csv": str(total_export_path.resolve()),
        "backtest_csv": str(backtest_path.resolve()),
        "day_compare_csv": str(selected_day_path.resolve()),
        "history_keys": list(history.keys()) if isinstance(history, dict) else [],
        "day_summary": day_summary,
    }

    summary_path = output_dir / "run_summary_torch.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log_block("RESULTS - NOTEBOOK-ALIGNED MODEL METRICS")
    for k, v in metrics_model.items():
        print(f"  {k}: {v}")

    log_block("RESULTS - WEEKLY NAIVE METRICS")
    for k, v in metrics_naive_weekly.items():
        print(f"  {k}: {v}")

    if metrics_old_legacy is not None:
        log_block("RESULTS - OLD LEGACY TOTAL FORECAST METRICS")
        for k, v in metrics_old_legacy.items():
            print(f"  {k}: {v}")

    if metrics_model_on_old_overlap is not None:
        log_block("RESULTS - NEW MODEL METRICS ON EXACT OLD OVERLAP")
        for k, v in metrics_model_on_old_overlap.items():
            print(f"  {k}: {v}")

    if comparison_new_vs_old is not None:
        log_block("RESULTS - NEW VS OLD LEGACY")
        print(f"  overlap_count: {old_overlap_count}")
        for k, v in comparison_new_vs_old.items():
            print(f"  {k}: {v}")

    log_block("RESULTS - COMPARISON")
    for k, v in comparison.items():
        print(f"  {k}: {v}")

    log_block("OUTPUT FILES")
    print(f"notebook legacy-schema csv: {notebook_export_path.resolve()}")
    print(f"total forecast csv:         {total_export_path.resolve()}")
    print(f"backtest csv:               {backtest_path.resolve()}")
    print(f"selected-day csv:           {selected_day_path.resolve()}")
    print(f"summary json:               {summary_path.resolve()}")
    print(f"saved model:                {Path(model_path).resolve()}")


if __name__ == "__main__":
    main()
