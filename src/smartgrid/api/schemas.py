from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ConsumptionPredictRequest(BaseModel):
    features: dict[str, float]


class ConsumptionPredictResponse(BaseModel):
    prediction: float
    model_type: str
    feature_columns: list[str]


class ForecastRuntimeRequest(BaseModel):
    dataset_key: str | None = None
    catalog_path: str | None = None
    historical_csv: str | None = None
    weather_csv: str | None = None
    holidays_xlsx: str | None = None
    benchmark_csv: str | None = None
    current_dir: str | None = None
    device: str = "auto"
    allow_fallback: bool = False


class ConsumptionNextDayRequest(ForecastRuntimeRequest):
    pass


class ConsumptionByDateRequest(ForecastRuntimeRequest):
    target_date: str


class ConsumptionReplayRequest(ForecastRuntimeRequest):
    start_date: str
    end_date: str


class ForecastPoint(BaseModel):
    Date: str
    Ptot_TOTAL_Forecast: float
    Ptot_TOTAL_Real: float | None = None


class ConsumptionForecastResponse(BaseModel):
    target_date: str
    points: list[ForecastPoint]
    model_run_id: str
    dataset_key: str | None = None
    forecast_mode: str


class ConsumptionReplayResponse(BaseModel):
    start_date: str
    end_date: str
    points: list[ForecastPoint]
    requested_model_run_id: str
    effective_model_run_ids: list[str]
    dataset_key: str | None = None
    forecast_mode: str
    overall_metrics: dict[str, Any] | None = None


class ConsumptionModelInfoResponse(BaseModel):
    model_config: dict[str, Any]
    latest_summary: dict[str, Any] | None = None
    dataset_key: str | None = None
    forecast_mode: str | None = None
