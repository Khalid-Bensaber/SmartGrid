from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI, HTTPException

from smartgrid.api.jobs import job_manager
from smartgrid.api.schemas import (
    ConsumptionBenchmarkFeaturesRequest,
    ConsumptionBenchmarkFeaturesResponse,
    ConsumptionBenchmarkReplayRequest,
    ConsumptionBenchmarkReplayResponse,
    ConsumptionByDateRequest,
    ConsumptionForecastResponse,
    ConsumptionModelInfoResponse,
    ConsumptionModelListResponse,
    ConsumptionNextDayRequest,
    ConsumptionPredictRequest,
    ConsumptionPredictResponse,
    ConsumptionPromoteRequest,
    ConsumptionPromoteResponse,
    ConsumptionReplayRequest,
    ConsumptionReplayResponse,
    ConsumptionTrainRequest,
    ConsumptionTrainResponse,
    JobResultResponse,
    JobStatusResponse,
    JobSubmissionResponse,
)
from smartgrid.api.services import (
    ApiServiceError,
    get_consumption_model_info,
    list_consumption_models,
    predict_consumption_from_features_service,
    run_consumption_feature_benchmark,
    run_consumption_forecast,
    run_consumption_promote,
    run_consumption_replay,
    run_consumption_replay_benchmark,
    run_consumption_training,
)

app = FastAPI(title="SmartGrid API", version="0.3.0")


def _raise_http(exc: Exception) -> None:
    if isinstance(exc, ApiServiceError):
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError | ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


def _submit_job(job_type: str, fn: Callable[..., dict[str, Any]], **kwargs) -> JobSubmissionResponse:
    record = job_manager.submit(job_type, fn, **kwargs)
    return JobSubmissionResponse(**record.as_dict(include_result=False))


def _get_job_or_404(job_id: str):
    record = job_manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return record


@app.get("/")
def root():
    return {
        "message": "SmartGrid API running",
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/jobs", response_model=list[JobStatusResponse])
def list_jobs():
    return job_manager.list()


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    record = _get_job_or_404(job_id)
    return record.as_dict(include_result=False)


@app.get("/jobs/{job_id}/result", response_model=JobResultResponse)
def get_job_result(job_id: str):
    record = _get_job_or_404(job_id)
    return record.as_dict(include_result=True)


@app.get("/consumption/model-info", response_model=ConsumptionModelInfoResponse)
def consumption_model_info(current_dir: str | None = None, device: str = "auto"):
    try:
        return get_consumption_model_info(current_dir=current_dir, device=device)
    except Exception as exc:
        _raise_http(exc)


@app.get("/consumption/models", response_model=ConsumptionModelListResponse)
def consumption_models(
    current_dir: str | None = None,
    artifacts_root: str = "artifacts",
    benchmark_csv: str | None = None,
):
    try:
        return list_consumption_models(
            current_dir=current_dir,
            artifacts_root=artifacts_root,
            benchmark_csv=benchmark_csv,
        )
    except Exception as exc:
        _raise_http(exc)


@app.post("/consumption/forecast/next-day", response_model=ConsumptionForecastResponse)
def consumption_forecast_next_day(payload: ConsumptionNextDayRequest):
    try:
        return run_consumption_forecast(**payload.model_dump())
    except Exception as exc:
        _raise_http(exc)


@app.post("/consumption/forecast/by-date", response_model=ConsumptionForecastResponse)
def consumption_forecast_by_date(payload: ConsumptionByDateRequest):
    try:
        return run_consumption_forecast(**payload.model_dump())
    except Exception as exc:
        _raise_http(exc)


@app.post("/consumption/replay", response_model=ConsumptionReplayResponse)
def consumption_replay(payload: ConsumptionReplayRequest):
    try:
        return run_consumption_replay(**payload.model_dump())
    except Exception as exc:
        _raise_http(exc)


@app.post("/consumption/replay/async", response_model=JobSubmissionResponse)
def consumption_replay_async(payload: ConsumptionReplayRequest):
    return _submit_job("consumption_replay", run_consumption_replay, **payload.model_dump())


@app.post("/consumption/predict-from-features", response_model=ConsumptionPredictResponse)
def predict_consumption_from_features(payload: ConsumptionPredictRequest):
    try:
        return predict_consumption_from_features_service(**payload.model_dump())
    except Exception as exc:
        _raise_http(exc)


@app.post("/consumption/train", response_model=ConsumptionTrainResponse)
def consumption_train(payload: ConsumptionTrainRequest):
    try:
        return run_consumption_training(**payload.model_dump())
    except Exception as exc:
        _raise_http(exc)


@app.post("/consumption/train/async", response_model=JobSubmissionResponse)
def consumption_train_async(payload: ConsumptionTrainRequest):
    return _submit_job("consumption_train", run_consumption_training, **payload.model_dump())


@app.post("/consumption/promote", response_model=ConsumptionPromoteResponse)
def consumption_promote(payload: ConsumptionPromoteRequest):
    try:
        return run_consumption_promote(**payload.model_dump())
    except Exception as exc:
        _raise_http(exc)


@app.post("/consumption/promote/async", response_model=JobSubmissionResponse)
def consumption_promote_async(payload: ConsumptionPromoteRequest):
    return _submit_job("consumption_promote", run_consumption_promote, **payload.model_dump())


@app.post(
    "/consumption/benchmark/replay",
    response_model=ConsumptionBenchmarkReplayResponse,
)
def consumption_benchmark_replay(payload: ConsumptionBenchmarkReplayRequest):
    try:
        return run_consumption_replay_benchmark(**payload.model_dump())
    except Exception as exc:
        _raise_http(exc)


@app.post("/consumption/benchmark/replay/async", response_model=JobSubmissionResponse)
def consumption_benchmark_replay_async(payload: ConsumptionBenchmarkReplayRequest):
    return _submit_job(
        "consumption_benchmark_replay",
        run_consumption_replay_benchmark,
        **payload.model_dump(),
    )


@app.post(
    "/consumption/benchmark/features",
    response_model=ConsumptionBenchmarkFeaturesResponse,
)
def consumption_benchmark_features(payload: ConsumptionBenchmarkFeaturesRequest):
    try:
        return run_consumption_feature_benchmark(**payload.model_dump())
    except Exception as exc:
        _raise_http(exc)


@app.post("/consumption/benchmark/features/async", response_model=JobSubmissionResponse)
def consumption_benchmark_features_async(payload: ConsumptionBenchmarkFeaturesRequest):
    return _submit_job(
        "consumption_benchmark_features",
        run_consumption_feature_benchmark,
        **payload.model_dump(),
    )
