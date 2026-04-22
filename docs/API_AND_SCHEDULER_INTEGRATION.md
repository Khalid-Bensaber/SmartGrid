# API and Scheduler Integration

Audience: backend integrators, orchestration engineers, and evaluators who want to drive SmartGrid through HTTP instead of only local CLI commands.

The API mirrors the same consumption workflow exposed by the CLI: inspect models, train, promote, forecast, replay, and benchmark.

## Why The API Exists

The FastAPI layer lets an external system trigger the same project capabilities behind stable HTTP endpoints:

- inspect the promoted model
- train or promote runs
- forecast one day
- replay historical periods
- benchmark runs or configs
- poll background jobs

This is useful for schedulers, demo dashboards, or automation agents that should not shell directly into the repository.

## Route Categories

### Health And Jobs

- `GET /`
- `GET /health`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`

### Model Inspection

- `GET /consumption/model-info`
- `GET /consumption/models`

### Forecasting

- `POST /consumption/forecast/next-day`
- `POST /consumption/forecast/by-date`
- `POST /consumption/predict-from-features`

### Replay

- `POST /consumption/replay`
- `POST /consumption/replay/async`

### Training And Promotion

- `POST /consumption/train`
- `POST /consumption/train/async`
- `POST /consumption/promote`
- `POST /consumption/promote/async`

### Benchmarks

- `POST /consumption/benchmark/replay`
- `POST /consumption/benchmark/replay/async`
- `POST /consumption/benchmark/features`
- `POST /consumption/benchmark/features/async`

## Sync Vs Async Usage

Use sync endpoints when:

- the work is short
- you want the full response in one call
- you are scripting a local demo or notebook

Use async endpoints when:

- training may take time
- replay spans a long date range
- benchmark requests fan out across many runs or configs

Current split:

- forecasting endpoints are sync only
- replay, training, promotion, and benchmark routes also have async variants

## Typical Forecast Payloads

Forecast the next available day after the latest timestamp in history:

```bash
curl -X POST http://localhost:8000/consumption/forecast/next-day \
  -H "Content-Type: application/json" \
  -d '{}'
```

Forecast an explicit day:

```bash
curl -X POST http://localhost:8000/consumption/forecast/by-date \
  -H "Content-Type: application/json" \
  -d '{
    "target_date": "2026-01-15",
    "dataset_key": "full_2020_2026",
    "write_outputs": true
  }'
```

Important response fields:

- `target_date`
- `points`
- `model_run_id`
- `requested_model_run_id`
- `fallback_used`
- output CSV paths when `write_outputs=true`

## Typical Replay Payload

```bash
curl -X POST http://localhost:8000/consumption/replay \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2026-01-01",
    "end_date": "2026-01-31",
    "dataset_key": "full_2020_2026",
    "write_outputs": true
  }'
```

Important response fields:

- `overall_metrics`
- `skipped_days`
- `effective_model_run_ids`
- `n_requested_days`
- `n_forecasted_days`
- `n_skipped_days`
- replay output paths

## Async Job Polling Flow

Example async replay request:

```bash
curl -X POST http://localhost:8000/consumption/replay/async \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2026-01-01",
    "end_date": "2026-01-15",
    "dataset_key": "full_2020_2026",
    "write_outputs": true
  }'
```

The API returns a `job_id`.

Poll status:

```bash
curl http://localhost:8000/jobs/<job_id>
```

Fetch the final payload:

```bash
curl http://localhost:8000/jobs/<job_id>/result
```

Status lifecycle:

- `queued`
- `running`
- `succeeded`
- `failed`

## Scheduler Integration Patterns

### Daily Forecast Pattern

1. Ensure a promoted model exists.
2. Call `POST /consumption/forecast/next-day` or `/forecast/by-date`.
3. Persist the returned CSV path and metadata in the scheduler’s own store.
4. Notify downstream systems.

### Periodic Retraining Pattern

1. Call `POST /consumption/train/async`.
2. Poll the returned job until it succeeds.
3. Inspect replay or offline metrics according to your acceptance rules.
4. Call `POST /consumption/promote` only after the run is approved.
5. Resume daily forecasting against the new current bundle.

### Evaluation Campaign Pattern

1. Call `POST /consumption/benchmark/replay` or the async variant.
2. Store the returned summary CSV and manifest JSON.
3. Use those outputs to decide whether a promoted bundle should change.

## Current In-Memory Job Limitation

The async job manager in `src/smartgrid/api/jobs.py` is:

- process-local
- in-memory
- backed by a thread pool

Implications:

- job state is lost when the API process restarts
- job state is not shared across multiple API workers
- this is suitable for demos and light orchestration, not durable distributed scheduling

If you need durable orchestration later, keep the HTTP contracts and replace the job backend behind them.

## Additional Integration Notes

- The API guards access to the promoted `current` bundle with in-process locks. That protects concurrent reads and promotions inside one process only.
- The same strict day-ahead semantics as the CLI apply here. Replay remains the official runtime benchmark.
- Most request models accept dataset overrides such as `dataset_key`, `historical_csv`, `weather_csv`, and `holidays_xlsx`. Use those only when you truly need to bypass the default catalog resolution.

## When To Prefer The CLI Instead

Use the CLI when:

- you are developing locally
- you want immediate terminal logs
- you are iterating in notebooks on the same machine
- you do not need an HTTP boundary

Use the API when:

- another service owns orchestration
- you need polling semantics
- you want to decouple the scheduler from direct shell access

## Read Next

- [Operations and Deployment](OPERATIONS_AND_DEPLOYMENT.md)
- [Maintainer Guide](../MAINTAINER_GUIDE.md)
