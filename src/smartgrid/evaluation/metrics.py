from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from smartgrid.common.constants import TOL_ABS, TOL_REL


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


def build_metrics_df(merged: pd.DataFrame, real_col: str, fc_col: str, tol_abs: float = TOL_ABS, tol_rel: float = TOL_REL):
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


def seasonal_naive_weekly(real: pd.Series, lag: str = "7D") -> pd.Series:
    lag_delta = pd.Timedelta(lag)
    values = real.reindex(real.index - lag_delta).to_numpy()
    return pd.Series(values, index=real.index)


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
