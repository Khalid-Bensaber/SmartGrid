from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from smartgrid.api.schemas import ConsumptionPredictRequest, ConsumptionPredictResponse
from smartgrid.common.utils import get_device
from smartgrid.inference.consumption import predict_from_feature_dict
from smartgrid.registry.model_registry import load_current_consumption_bundle

app = FastAPI(title="SmartGrid API", version="0.2.0")

REGISTRY_CURRENT_DIR = Path("artifacts/models/consumption/current")


@app.get("/")
def root():
    return {"message": "SmartGrid API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/consumption/model-info")
def consumption_model_info():
    if not REGISTRY_CURRENT_DIR.exists() or not (REGISTRY_CURRENT_DIR / "model.pt").exists():
        raise HTTPException(status_code=404, detail="No promoted consumption model found")
    device = get_device("auto")
    _, _, _, model_config, summary = load_current_consumption_bundle(REGISTRY_CURRENT_DIR, device)
    return {
        "model_config": model_config,
        "latest_summary": summary,
    }


@app.post("/consumption/predict-from-features", response_model=ConsumptionPredictResponse)
def predict_consumption_from_features(payload: ConsumptionPredictRequest):
    if not REGISTRY_CURRENT_DIR.exists() or not (REGISTRY_CURRENT_DIR / "model.pt").exists():
        raise HTTPException(status_code=404, detail="No promoted consumption model found")

    device = get_device("auto")
    model, x_scaler, y_scaler, model_config, summary = load_current_consumption_bundle(REGISTRY_CURRENT_DIR, device)
    feature_columns = summary["feature_columns"] if summary else model_config.get("feature_columns")
    if feature_columns is None:
        raise HTTPException(status_code=500, detail="Feature columns missing from summary/model config")

    missing = [column for column in feature_columns if column not in payload.features]
    if missing:
        raise HTTPException(status_code=400, detail={"missing_feature_columns": missing})

    prediction = predict_from_feature_dict(model, x_scaler, y_scaler, payload.features, feature_columns, device)
    return ConsumptionPredictResponse(
        prediction=prediction,
        model_type=model_config.get("model_type", "unknown"),
        feature_columns=feature_columns,
    )
