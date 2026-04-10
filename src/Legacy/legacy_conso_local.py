#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Legacy local consumption model runner - improved single-file version.

Goals:
- run locally from clean CSV files
- rebuild total consumption from building columns
- reproduce legacy-style manual lag features
- train a dense MLP (TensorFlow preferred, sklearn fallback)
- support GPU if TensorFlow sees it
- save / resume model + scalers
- perform chronological backtest
- export one selected day for notebook comparison
- compare against old legacy forecast CSV using the SAME total definition as the notebook
  (sum of 4 building forecast columns; Ptot_Ilot_Forecast is intentionally ignored)
- compute notebook-aligned metrics

Project layout expected:
smart-grid/
├── data/
│   └── processed/
│       ├── Holidays.xlsx
│       ├── 2025-12-02_10_15_previsions_data_conso_historiques_clean.csv
│       └── conso/
│           └── 2025-12-02_10_15_previsions_data_conso_prev_ptot_clean.csv
└── src/
    └── Legacy/
        └── legacy_conso_local.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# -------------------------------
# Constants
# -------------------------------
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


# -------------------------------
# CLI
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Improved legacy local consumption trainer/backtester")
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
    p.add_argument("--output-dir", type=str, default="artifacts/legacy/consumption")
    p.add_argument("--output-filename", type=str, default="legacy_local_prev_ptot_clean.csv")
    p.add_argument("--date-col", type=str, default="Date")

    # split controls
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--train-end-date", type=str, default=None, help="Optional explicit train end date YYYY-MM-DD")
    p.add_argument("--val-end-date", type=str, default=None, help="Optional explicit val end date YYYY-MM-DD")

    # model / training controls
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "tensorflow", "sklearn"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=720)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--hidden-layers", type=str, default="1024,512")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--verbose-fit", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--resume-model", type=str, default=None)
    p.add_argument("--skip-train", action="store_true", help="Load a model and scalers from output dir without retraining")

    # analysis / export
    p.add_argument("--analysis-date", type=str, default=None, help="Day to export/analyse (YYYY-MM-DD)")
    p.add_argument("--analysis-days", type=int, default=1, help="Number of consecutive days to export starting from analysis-date")

    return p.parse_args()


# -------------------------------
# Utils
# -------------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


def parse_hidden_layers(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def maybe_enable_tf_gpu() -> dict:
    info = {
        "tf_available": TF_AVAILABLE,
        "gpu_detected": False,
        "gpu_names": [],
    }
    if not TF_AVAILABLE:
        return info

    try:
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu_detected"] = len(gpus) > 0
        info["gpu_names"] = [str(g) for g in gpus]
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    except Exception:
        pass
    return info


def log_block(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# -------------------------------
# Data loading / feature engineering
# -------------------------------
def load_holiday_sets(holidays_xlsx: str) -> tuple[set, set]:
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
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    missing_cols = [c for c in TOTAL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required history columns: {missing_cols}")

    df[DEFAULT_TARGET_NAME] = df[TOTAL_COLUMNS].sum(axis=1)

    # Keep historical clean data as-is; only add default Airtemp if not present.
    if "Airtemp" not in df.columns:
        if "AirTemp" in df.columns:
            df["Airtemp"] = df["AirTemp"]
        else:
            df["Airtemp"] = DEFAULT_AIRTEMP_VALUE

    return df


def add_calendar_features(df: pd.DataFrame, holiday_dates: set, special_dates: set, date_col: str) -> pd.DataFrame:
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
    # same convention as legacy: special day overrides holiday flag
    out.loc[out["is_special_day"] == 1, "is_holiday"] = 0
    return out


def add_manual_lag_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
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
    df = df.dropna(subset=feature_cols + [DEFAULT_TARGET_NAME]).reset_index(drop=True)
    return df, feature_cols


# -------------------------------
# Splits
# -------------------------------
def chronological_split_by_ratio(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
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
        raise RuntimeError(
            "Date-based split produced an empty split. Check --train-end-date / --val-end-date."
        )
    return train_df, val_df, test_df


def make_splits(df: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.train_end_date and args.val_end_date:
        return chronological_split_by_dates(df, args.date_col, args.train_end_date, args.val_end_date)
    return chronological_split_by_ratio(df, train_ratio=args.train_ratio, val_ratio=args.val_ratio)


# -------------------------------
# TF callbacks / model
# -------------------------------
class EpochTimerCallback(tf.keras.callbacks.Callback if TF_AVAILABLE else object):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.train_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)
        avg_epoch = sum(self.epoch_times) / len(self.epoch_times)
        total_epochs = self.params.get("epochs", 0)
        done_epochs = epoch + 1
        remaining_epochs = total_epochs - done_epochs
        eta_sec = remaining_epochs * avg_epoch

        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        mae = logs.get("mae")
        val_mae = logs.get("val_mae")

        def f(x):
            return f"{x:.6f}" if x is not None else "nan"

        print(
            f"[Epoch {done_epochs}/{total_epochs}] "
            f"loss={f(loss)} val_loss={f(val_loss)} "
            f"mae={f(mae)} val_mae={f(val_mae)} "
            f"time={epoch_time:.1f}s eta~={eta_sec/60:.1f} min"
        )


def make_tf_model(input_dim: int, hidden_layers: Iterable[int], learning_rate: float):
    model = tf.keras.Sequential(name="legacy_dense_mlp")
    first = True
    for units in hidden_layers:
        if first:
            model.add(tf.keras.layers.Dense(units, input_dim=input_dim, activation="relu"))
            first = False
        else:
            model.add(tf.keras.layers.Dense(units, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="linear"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae"])
    return model


def pick_backend(requested: str) -> str:
    if requested == "tensorflow":
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow backend requested but tensorflow is not installed.")
        return "tensorflow"
    if requested == "sklearn":
        return "sklearn"
    return "tensorflow" if TF_AVAILABLE else "sklearn"


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    backend: str,
    epochs: int,
    batch_size: int,
    seed: int,
    hidden_layers: tuple[int, ...],
    learning_rate: float,
    resume_model: str | None = None,
    patience: int = 8,
    verbose_fit: int = 0,
):
    if backend == "tensorflow":
        tf.keras.backend.clear_session()
        if resume_model:
            model = tf.keras.models.load_model(resume_model)
            print(f"[INFO] resumed TensorFlow model from: {resume_model}")
        else:
            model = make_tf_model(X_train.shape[1], hidden_layers=hidden_layers, learning_rate=learning_rate)

        print("\n[MODEL] TensorFlow model summary:")
        model.summary(print_fn=lambda x: print("  " + x))

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                mode="min",
                restore_best_weights=True,
            ),
            EpochTimerCallback(),
        ]

        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose_fit,
        )
        return model, history.history

    print("\n[MODEL] sklearn MLPRegressor")
    print(f"  hidden_layers={hidden_layers}")
    print(f"  learning_rate={learning_rate}")
    print(f"  max_iter={epochs}")
    print(f"  batch_size={batch_size}")

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        early_stopping=True,
        validation_fraction=0.1,
        max_iter=epochs,
        batch_size=batch_size if batch_size != "auto" else "auto",
        learning_rate_init=learning_rate,
        random_state=seed,
        verbose=True,
    )
    model.fit(X_train, y_train.ravel())
    history = {
        "loss_curve": getattr(model, "loss_curve_", []),
        "best_validation_score": getattr(model, "best_validation_score_", None),
    }
    return model, history


def predict_model(model, X_scaled, y_scaler, backend: str):
    if backend == "tensorflow":
        pred_scaled = model.predict(X_scaled, verbose=0)
    else:
        pred_scaled = model.predict(X_scaled).reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_scaled).ravel()
    return pred


# -------------------------------
# Metrics (aligned with notebook)
# -------------------------------
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
    derr = (dfc - dr)
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


# -------------------------------
# Benchmark / export helpers
# -------------------------------
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
    # Keep Ptot_Ilot_Forecast if present, but we DO NOT use it in metrics.
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
    # Keep legacy schema for quick notebook ingestion; do not rely on Ptot_Ilot in metrics.
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


def save_model_artifacts(model, x_scaler, y_scaler, backend: str, output_dir: Path):
    if backend == "tensorflow":
        model_path = output_dir / "legacy_model.keras"
        model.save(model_path)
    else:
        model_path = output_dir / "legacy_model_sklearn.joblib"
        joblib.dump(model, model_path)

    x_scaler_path = output_dir / "x_scaler.save"
    y_scaler_path = output_dir / "y_scaler.save"
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)
    return model_path, x_scaler_path, y_scaler_path


def load_saved_artifacts(output_dir: Path, backend: str):
    x_scaler_path = output_dir / "x_scaler.save"
    y_scaler_path = output_dir / "y_scaler.save"
    if backend == "tensorflow":
        model_path = output_dir / "legacy_model.keras"
        model = tf.keras.models.load_model(model_path)
    else:
        model_path = output_dir / "legacy_model_sklearn.joblib"
        model = joblib.load(model_path)
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    return model, x_scaler, y_scaler, model_path


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


# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tf_gpu_info = maybe_enable_tf_gpu()
    backend = pick_backend(args.backend)

    log_block("ENVIRONMENT")
    print(f"[INFO] backend selected:          {backend}")
    print(f"[INFO] tensorflow available:      {tf_gpu_info['tf_available']}")
    print(f"[INFO] tensorflow GPU detected:   {tf_gpu_info['gpu_detected']}")
    if tf_gpu_info["gpu_names"]:
        print(f"[INFO] tensorflow GPU devices:    {tf_gpu_info['gpu_names']}")

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

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    X_val_scaled = x_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val)
    X_test_scaled = x_scaler.transform(X_test)

    log_block("TRAINING / LOADING MODEL")
    if args.skip_train:
        print("[INFO] skip-train enabled -> loading model + scalers from output dir")
        model, x_scaler, y_scaler, model_path = load_saved_artifacts(output_dir, backend)
        history = {"loaded_from_disk": True}
    else:
        train_start = time.time()
        model, history = train_model(
            X_train_scaled,
            y_train_scaled,
            X_val_scaled,
            y_val_scaled,
            backend=backend,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            hidden_layers=hidden_layers,
            learning_rate=args.learning_rate,
            resume_model=args.resume_model,
            patience=args.patience,
            verbose_fit=args.verbose_fit,
        )
        train_duration_sec = time.time() - train_start
        print(f"[INFO] training duration: {train_duration_sec/60:.2f} min")
        model_path, _, _ = save_model_artifacts(model, x_scaler, y_scaler, backend, output_dir)

    log_block("BACKTEST")
    y_pred = predict_model(model, X_test_scaled, y_scaler, backend=backend)
    basic_metrics = compute_basic_metrics(y_test, y_pred)
    for k, v in basic_metrics.items():
        print(f"[INFO] basic {k}: {v}")

    backtest = test_df[[args.date_col, DEFAULT_TARGET_NAME, "lag_d1"]].copy()
    backtest["Ptot_TOTAL_Forecast"] = y_pred
    backtest["name"] = "CONSO_Prevision_Data"

    benchmark = load_old_benchmark(args.benchmark_csv, date_col=args.date_col)
    if benchmark is not None:
        backtest = backtest.merge(benchmark, on=args.date_col, how="left")

    # notebook-aligned metrics: model vs weekly naive vs old legacy total forecast
    model_eval = build_metrics_df(
        merged=backtest[[args.date_col, DEFAULT_TARGET_NAME, "Ptot_TOTAL_Forecast"]].set_index(args.date_col),
        real_col=DEFAULT_TARGET_NAME,
        fc_col="Ptot_TOTAL_Forecast",
        tol_abs=TOL_ABS,
        tol_rel=TOL_REL,
    )

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
        "backend": backend,
        "tf_gpu_info": tf_gpu_info,
        "feature_columns": feature_cols,
        "hidden_layers": list(hidden_layers),
        "learning_rate": args.learning_rate,
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

    summary_path = output_dir / "run_summary.json"
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
