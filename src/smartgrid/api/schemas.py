from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobSubmissionResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    created_at: str


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class JobResultResponse(JobStatusResponse):
    result: dict[str, Any] | None = None


class ForecastPoint(BaseModel):
    Date: str
    Ptot_TOTAL_Forecast: float
    Ptot_TOTAL_Real: float | None = None


class ReplaySkippedDay(BaseModel):
    target_date: str
    reason: str


class ForecastRuntimeRequest(BaseModel):
    dataset_key: str | None = None
    catalog_path: str | None = None
    historical_csv: str | None = None
    weather_csv: str | None = None
    holidays_xlsx: str | None = None
    benchmark_csv: str | None = None
    current_dir: str | None = None
    artifacts_root: str = "artifacts"
    device: str = "auto"
    allow_fallback: bool = False


class ConsumptionNextDayRequest(ForecastRuntimeRequest):
    output_csv: str | None = None
    write_outputs: bool = True


class ConsumptionByDateRequest(ForecastRuntimeRequest):
    target_date: str
    output_csv: str | None = None
    write_outputs: bool = True


class ConsumptionReplayRequest(ForecastRuntimeRequest):
    start_date: str
    end_date: str
    write_per_day: bool = False
    write_outputs: bool = True


class ConsumptionForecastResponse(BaseModel):
    target_date: str
    points: list[ForecastPoint]
    model_run_id: str
    requested_model_run_id: str
    dataset_key: str | None = None
    forecast_mode: str
    fallback_used: bool = False
    current_output_csv: str | None = None
    archive_output_csv: str | None = None
    custom_output_csv: str | None = None


class ConsumptionReplayResponse(BaseModel):
    start_date: str
    end_date: str
    points: list[ForecastPoint]
    requested_model_run_id: str
    effective_model_run_ids: list[str]
    dataset_key: str | None = None
    forecast_mode: str
    overall_metrics: dict[str, Any] | None = None
    skipped_days: list[ReplaySkippedDay] = Field(default_factory=list)
    fallback_used: bool = False
    output_csv: str | None = None
    metrics_json: str | None = None
    per_day_dir: str | None = None
    n_days: int | None = None
    n_rows: int | None = None
    n_requested_days: int | None = None
    n_forecasted_days: int | None = None
    n_skipped_days: int | None = None
    per_day_metrics: list[dict[str, Any]] = Field(default_factory=list)
    per_day_model_usage: list[dict[str, Any]] = Field(default_factory=list)


class ConsumptionPredictRequest(BaseModel):
    features: dict[str, float]
    current_dir: str | None = None
    device: str = "auto"


class ConsumptionPredictResponse(BaseModel):
    prediction: float
    model_type: str
    feature_columns: list[str]
    model_run_id: str | None = None
    current_dir: str | None = None


class ConsumptionModelInfoResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_bundle_config: dict[str, Any] = Field(
        validation_alias="model_config",
        serialization_alias="model_config",
    )
    latest_summary: dict[str, Any] | None = None
    dataset_key: str | None = None
    forecast_mode: str | None = None
    model_run_id: str | None = None
    current_dir: str | None = None


class ConsumptionModelListResponse(BaseModel):
    current_dir: str
    artifacts_root: str
    models: list[dict[str, Any]]


class ConsumptionTrainRequest(BaseModel):
    config: str = "configs/consumption/mlp_strict_day_ahead_baseline.yaml"
    analysis_date: str | None = None
    analysis_days: int = 1
    resume_checkpoint: str | None = None
    dataset_key: str | None = None
    catalog_path: str | None = None
    historical_csv: str | None = None
    benchmark_csv: str | None = None
    weather_csv: str | None = None
    holidays_xlsx: str | None = None
    promote: bool = False
    profile: bool = False
    device: str | None = None


class ConsumptionTrainResponse(BaseModel):
    run_id: str
    run_dir: str
    exports_dir: str
    promoted: bool
    config: str
    experiment_name: str | None = None
    selected_analysis_day: str
    train_duration_sec: float
    n_features: int
    epochs_ran: int
    best_val_loss: float | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    n_train_rows: int
    n_val_rows: int
    n_test_rows: int
    feature_config: dict[str, Any]
    forecast_mode: str | None = None
    dataset_key: str | None = None
    device: str
    runtime_diagnostics: dict[str, Any]
    batching_strategy: str
    resident_data_bytes: int
    test_date_min: str
    test_date_max: str
    offline_test_metrics: dict[str, Any] | None = None
    metrics_model: dict[str, Any] | None = None
    profiling_enabled: bool


class ConsumptionPromoteRequest(BaseModel):
    run_id: str
    artifacts_root: str = "artifacts"


class ConsumptionPromoteResponse(BaseModel):
    run_id: str
    run_dir: str
    current_dir: str
    promoted: bool


class ConsumptionBenchmarkReplayRequest(BaseModel):
    model_refs: list[str]
    start_date: str
    end_date: str
    dataset_key: str | None = None
    catalog_path: str | None = None
    historical_csv: str | None = None
    weather_csv: str | None = None
    holidays_xlsx: str | None = None
    benchmark_csv: str | None = None
    artifacts_root: str = "artifacts"
    device: str = "auto"
    allow_fallback: bool = False


class ConsumptionBenchmarkReplayResponse(BaseModel):
    summary_csv: str
    manifest_json: str
    n_models: int
    start_date: str
    end_date: str
    rows: list[dict[str, Any]] = Field(default_factory=list)


class ConsumptionBenchmarkFeaturesRequest(BaseModel):
    configs: list[str]
    output_csv: str = "artifacts/benchmarks/consumption_feature_variants.csv"
    analysis_days: int = 1
    dataset_key: str | None = None
    catalog_path: str | None = None
    historical_csv: str | None = None
    weather_csv: str | None = None
    holidays_xlsx: str | None = None
    benchmark_csv: str | None = None
    replay_start_date: str | None = None
    replay_end_date: str | None = None


class ConsumptionBenchmarkFeaturesResponse(BaseModel):
    output_csv: str
    n_configs: int
    rows: list[dict[str, Any]] = Field(default_factory=list)
    runs: list[dict[str, Any]] = Field(default_factory=list)
