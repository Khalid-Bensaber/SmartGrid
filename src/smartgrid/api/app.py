print(">>> SMARTGRID API VERSION COMPLETE CHARGÉE <<<")
import os
print("FILE LOADED:", __file__)
print("WORKING DIR:", os.getcwd())
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from smartgrid.api.schemas import (
    ConsumptionByDateRequest,
    ConsumptionForecastResponse,
    ConsumptionModelInfoResponse,
    ConsumptionNextDayRequest,
    ConsumptionPredictRequest,
    ConsumptionPredictResponse,
    ConsumptionReplayRequest,
    ConsumptionReplayResponse,
    ForecastPoint,
)
from smartgrid.common.utils import get_device
from smartgrid.evaluation.reporting import evaluate_forecast_frame
from smartgrid.inference.consumption import predict_from_feature_dict
from smartgrid.inference.day_ahead import (
    build_forecast_runtime,
    forecast_next_day,
    forecast_target_day,
    replay_forecast_period,
)
from smartgrid.registry.model_registry import load_current_consumption_bundle

app = FastAPI(title="SmartGrid API", version="0.2.0")

REGISTRY_CURRENT_DIR = Path("artifacts/models/consumption/current")


@app.get("/")
def root():
    return {"message": "SmartGrid API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


def _resolve_current_dir(current_dir: str | None) -> Path:
    return Path(current_dir) if current_dir else REGISTRY_CURRENT_DIR


def _build_runtime(payload) -> object:
    current_dir = _resolve_current_dir(getattr(payload, "current_dir", None))
    if not current_dir.exists() or not (current_dir / "model.pt").exists():
        raise HTTPException(status_code=404, detail="No promoted consumption model found")

    try:
        return build_forecast_runtime(
            current_dir=current_dir,
            dataset_key=getattr(payload, "dataset_key", None),
            catalog_path=getattr(payload, "catalog_path", None),
            historical_csv=getattr(payload, "historical_csv", None),
            weather_csv=getattr(payload, "weather_csv", None),
            holidays_xlsx=getattr(payload, "holidays_xlsx", None),
            benchmark_csv=getattr(payload, "benchmark_csv", None),
            device_request=getattr(payload, "device", "auto"),
            allow_fallback=getattr(payload, "allow_fallback", False),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _frame_to_points(frame) -> list[ForecastPoint]:
    points = []
    for record in frame.to_dict(orient="records"):
        points.append(
            ForecastPoint(
                Date=str(record["Date"]),
                Ptot_TOTAL_Forecast=float(record["Ptot_TOTAL_Forecast"]),
                Ptot_TOTAL_Real=None
                if pd.isna(record.get("Ptot_TOTAL_Real"))
                else float(record["Ptot_TOTAL_Real"]),
            )
        )
    return points


@app.get("/consumption/model-info", response_model=ConsumptionModelInfoResponse)
def consumption_model_info():
    if not REGISTRY_CURRENT_DIR.exists() or not (REGISTRY_CURRENT_DIR / "model.pt").exists():
        raise HTTPException(status_code=404, detail="No promoted consumption model found")
    device = get_device("auto")
    bundle = load_current_consumption_bundle(REGISTRY_CURRENT_DIR, device)
    summary = bundle.summary or {}
    return ConsumptionModelInfoResponse(
        model_config=bundle.model_config,
        latest_summary=summary,
        dataset_key=summary.get("dataset_key"),
        forecast_mode=(summary.get("feature_config") or {}).get("forecast_mode"),
    )


@app.post("/consumption/forecast/next-day", response_model=ConsumptionForecastResponse)
def consumption_forecast_next_day(payload: ConsumptionNextDayRequest):
    runtime = _build_runtime(payload)
    try:
        forecast_df = forecast_next_day(runtime)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    target_date = str(forecast_df["target_date"].iloc[0]) if not forecast_df.empty else ""
    model_run_id = str(forecast_df["model_run_id"].iloc[0]) if not forecast_df.empty else "unknown"
    return ConsumptionForecastResponse(
        target_date=target_date,
        points=_frame_to_points(forecast_df),
        model_run_id=model_run_id,
        dataset_key=runtime.data_config.get("dataset_key"),
        forecast_mode=runtime.forecast_mode,
    )


@app.post("/consumption/forecast/by-date", response_model=ConsumptionForecastResponse)
def consumption_forecast_by_date(payload: ConsumptionByDateRequest):
    runtime = _build_runtime(payload)
    try:
        forecast_df = forecast_target_day(runtime, payload.target_date)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model_run_id = str(forecast_df["model_run_id"].iloc[0]) if not forecast_df.empty else "unknown"
    return ConsumptionForecastResponse(
        target_date=str(payload.target_date),
        points=_frame_to_points(forecast_df),
        model_run_id=model_run_id,
        dataset_key=runtime.data_config.get("dataset_key"),
        forecast_mode=runtime.forecast_mode,
    )


@app.post("/consumption/replay", response_model=ConsumptionReplayResponse)
def consumption_replay(payload: ConsumptionReplayRequest):
    runtime = _build_runtime(payload)
    try:
        replay = replay_forecast_period(runtime, payload.start_date, payload.end_date)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ConsumptionReplayResponse(
        start_date=replay.start_date,
        end_date=replay.end_date,
        points=_frame_to_points(replay.replay_df),
        requested_model_run_id=replay.requested_model_run_id,
        effective_model_run_ids=replay.effective_model_run_ids,
        dataset_key=runtime.data_config.get("dataset_key"),
        forecast_mode=runtime.forecast_mode,
        overall_metrics=evaluate_forecast_frame(replay.replay_df),
    )


@app.post("/consumption/predict-from-features", response_model=ConsumptionPredictResponse)
def predict_consumption_from_features(payload: ConsumptionPredictRequest):
    if not REGISTRY_CURRENT_DIR.exists() or not (REGISTRY_CURRENT_DIR / "model.pt").exists():
        raise HTTPException(status_code=404, detail="No promoted consumption model found")

    device = get_device("auto")
    bundle = load_current_consumption_bundle(REGISTRY_CURRENT_DIR, device)
    feature_columns = (
        bundle.summary["feature_columns"]
        if bundle.summary
        else bundle.model_config.get("feature_columns")
    )
    if feature_columns is None:
        raise HTTPException(
            status_code=500,
            detail="Feature columns missing from summary/model config",
        )

    missing = [column for column in feature_columns if column not in payload.features]
    if missing:
        raise HTTPException(status_code=400, detail={"missing_feature_columns": missing})

    prediction = predict_from_feature_dict(
        bundle.model,
        bundle.x_scaler,
        bundle.y_scaler,
        payload.features,
        feature_columns,
        device,
    )
    return ConsumptionPredictResponse(
        prediction=prediction,
        model_type=bundle.model_config.get("model_type", "unknown"),
        feature_columns=feature_columns,
    )
