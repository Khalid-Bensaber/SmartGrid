from __future__ import annotations

from pydantic import BaseModel


class ConsumptionPredictRequest(BaseModel):
    features: dict[str, float]


class ConsumptionPredictResponse(BaseModel):
    prediction: float
    model_type: str
    feature_columns: list[str]
