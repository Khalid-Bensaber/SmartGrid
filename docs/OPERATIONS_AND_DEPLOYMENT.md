# Operations and Deployment

Audience: operators, maintainers, and anyone running SmartGrid repeatedly for demos or evaluation.

This guide covers local execution, Docker usage, command reference, runtime outputs, and troubleshooting. It does not try to explain code internals in detail.

## Operating Modes

SmartGrid can be used in four aligned ways:

- `make` targets for everyday local work
- direct `uv run python scripts/...` commands when you need explicit arguments
- FastAPI for HTTP orchestration
- Docker Compose for reproducible demo and pre-deployment environments

All of these surfaces target the same consumption pipeline and the same promoted bundle convention.

## Docker Files And Compose Layout

Relevant files:

- `Dockerfile`
- `docker-compose.yml`
- `docker-compose.gpu.yml`
- `docker/entrypoint.sh`
- `.dockerignore`

Key behavior:

- One reusable image is built as `smartgrid:dev`
- `cli`, `api`, and `notebook` reuse that same image
- The repository is mounted into `/workspace`
- A named `uv` cache volume speeds up repeated starts
- The entrypoint creates runtime directories, runs `uv sync`, and executes `make doctor`

The Docker setup is intentionally development-oriented. It is designed to preserve local edits, local artifacts, and tracked processed data through the bind mount.

## CPU Vs GPU Paths

### Local Execution

- `device=auto` selects CUDA when available and CPU otherwise
- You can force CPU with CLI or API arguments such as `--device cpu`
- You can force CUDA with `--device cuda` when the host actually provides it

### Docker Execution

CPU or non-GPU host:

```bash
docker compose build cli --progress=plain
docker compose run --rm cli
docker compose up api
```

GPU host:

```bash
docker compose build cli --progress=plain
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm cli
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up api
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile notebook up notebook
```

Use the GPU override only on machines with an NVIDIA runtime. The main Compose file stays CPU-safe on purpose.

## Make Command Reference

### Environment And Quality

- `make install`: installs all dependency groups with `uv sync --all-groups`
- `make install-core`: installs only the runtime dependency set with `uv sync`
- `make install-dev`: installs the dev group with `uv sync --group dev`
- `make install-dev-legacy`: installs all groups, including legacy extras
- `make doctor`: runs `uv run python scripts/check_setup.py` to validate the dataset catalog paths
- `make lint`: runs Ruff on `src`, `tests`, and `scripts`
- `make test`: runs the pytest suite
- `make build`: builds the wheel and source distribution
- `make verify`: runs `doctor`, `test`, and `build` in sequence
- `make notebook`: starts JupyterLab locally

### Runtime Operations

- `make serve-api`: runs `uv run uvicorn smartgrid.api:app --reload --host 0.0.0.0 --port 8000`
- `make train-consumption`: runs `scripts/train_consumption.py` with the configured YAML, dataset key, and analysis window
- `make train-promote`: same as training, but adds `--promote` so the new bundle becomes current immediately
- `make promote-consumption RUN_ID=...`: copies an existing run bundle into `artifacts/models/consumption/current`
- `make predict-next-day`: runs day-ahead forecasting from the promoted bundle
- `make replay-period`: replays one historical interval day by day and evaluates forecast quality where truth exists
- `make benchmark-features`: trains several configs and ranks them by replay results
- `make benchmark-replay MODEL_REFS='run_a run_b'`: benchmarks existing runs on the same replay period

## Direct CLI Equivalents

- `make doctor` -> `uv run python scripts/check_setup.py`
- `make train-consumption` -> `uv run python scripts/train_consumption.py --config "$CONFIG" --analysis-days "$ANALYSIS_DAYS" --dataset-key "$DATASET_KEY"`
- `make train-promote` -> `uv run python scripts/train_consumption.py --config "$CONFIG" --analysis-days "$ANALYSIS_DAYS" --dataset-key "$DATASET_KEY" --promote`
- `make promote-consumption RUN_ID=...` -> `uv run python scripts/promote_consumption_run.py --run-id "$RUN_ID"`
- `make predict-next-day` -> `uv run python scripts/predict_next_day.py --dataset-key "$DATASET_KEY" --historical-csv "$HISTORICAL_CSV" --weather-csv "$WEATHER_CSV" --holidays-xlsx "$HOLIDAYS_XLSX" --target-date "$TARGET_DATE"`
- `make replay-period` -> `uv run python scripts/replay_period.py --dataset-key "$DATASET_KEY" --historical-csv "$HISTORICAL_CSV" --weather-csv "$WEATHER_CSV" --holidays-xlsx "$HOLIDAYS_XLSX" --start-date "$START_DATE" --end-date "$END_DATE"`
- `make serve-api` -> `uv run uvicorn smartgrid.api:app --reload --host "$API_HOST" --port "$API_PORT"`
- `make benchmark-features` -> `uv run python scripts/benchmark_feature_variants.py ...`
- `make benchmark-replay` -> `uv run python scripts/benchmark_replay_models.py --start-date "$START_DATE" --end-date "$END_DATE" ...`

## Operator Workflow

1. Run `make doctor` after a fresh clone or when datasets change.
2. Run `make verify` before a serious demo or handoff.
3. Train with `make train-promote` to refresh the promoted bundle.
4. Validate the runtime path with `make predict-next-day`.
5. Evaluate operational behavior with `make replay-period`.
6. Expose or integrate the workflow through `make serve-api` when needed.

If you want comparative evaluation rather than a single promoted model check, use `make benchmark-features` or `make benchmark-replay`.

## Artifacts And Logs

Generated outputs live under `artifacts/`.

- `artifacts/runs/consumption/<run_id>/`: persisted model bundle used for later promotion or replay
- `artifacts/exports/consumption/<run_id>/`: training exports, summaries, and offline diagnostics
- `artifacts/models/consumption/current/`: the promoted runtime bundle used by default forecast and API calls
- `artifacts/forecasts/consumption/current/`: latest forecast CSVs by target date
- `artifacts/forecasts/consumption/archive/<run_id>/`: archived forecast outputs by originating model
- `artifacts/replays/consumption/<stamp>__<start>__<end>/`: replay forecasts, metrics, and optional per-day files
- `artifacts/benchmarks/replay/...`: replay benchmark summaries across several runs
- `artifacts/notebook_exports/cli_demo_v4/`: notebook-oriented demo exports
- `artifacts/logs/train|predict|replay/`: execution logs for the main operational channels

## Troubleshooting

- `make doctor` reports missing files:
  The dataset catalog points to tracked files under `data/processed/`. Restore or pull the missing assets before training or replay.

- `make predict-next-day` fails because no model exists:
  Run `make train-promote` first, or promote an existing run with `make promote-consumption RUN_ID=...`.

- Replay skips some days:
  This is expected when target-day truth coverage is incomplete. The replay metrics JSON records skipped dates and reasons.

- Forecasting fails with missing features:
  The promoted bundle may require lag or exogenous values that are unavailable for the requested date. Check the log and consider retraining or using a compatible fallback bundle.

- API async jobs disappear after a restart:
  The job manager stores state in memory only. Restarting the API loses queued and completed job state.

- Docker starts slowly on the first run:
  The entrypoint resynchronizes dependencies with `uv sync`. This is normal and becomes faster once the cache is warm.

- CUDA is not detected:
  Use the GPU Compose override on supported machines, or force CPU explicitly instead of assuming CUDA exists.

## Deployment Caveats

- The current API job manager is suited to demos and single-process use, not durable distributed orchestration.
- The repo intentionally versions processed demo data under `data/processed/`. Do not hide that directory behind a separate Docker volume.
- The active production-style workflow is consumption forecasting. PV assets exist, but the API and CLI are not yet parallelized into a production pipeline.

## Read Next

- [Quick Start](QUICKSTART.md)
- [API and Scheduler Integration](API_AND_SCHEDULER_INTEGRATION.md)
- [Notebook and Demo Guide](NOTEBOOK_AND_DEMO_GUIDE.md)
