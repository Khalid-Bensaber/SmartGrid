# Smart Grid

Smart Grid is an end-to-end forecasting platform for electricity consumption. It combines a reproducible data pipeline, configurable feature engineering, PyTorch model training, model promotion, strict day-ahead inference, historical replay benchmarking, a FastAPI orchestration layer, notebooks, and Docker-based deployment.

The repository is designed to be usable in several ways:

- as a local Python project driven by `make`
- as a CLI toolkit driven by scripts and package entry points
- as an HTTP API that can orchestrate training, promotion, forecasting, replay, and benchmarking
- as a Dockerized demo and deployment environment for CLI, API, and notebooks

The current operationally exposed pipeline is the electricity consumption pipeline. The repository also contains photovoltaic-related data assets, but the productionized CLI and API surface described here focuses on consumption forecasting.

## What This Project Solves

This project answers a practical question:

How do you go from historical electricity consumption data to a model that can be trained, benchmarked, promoted, replayed on past periods, and served consistently through both CLI and API interfaces?

The repository provides that full chain:

- dataset resolution through a catalog
- feature generation with time-aware safeguards
- train/validation/test splitting by date
- PyTorch MLP training with configurable architectures
- artifact and run summary persistence
- promotion of a selected run to a "current" production-style bundle
- strict day-ahead inference from the promoted model
- replay over historical periods with evaluation metrics
- model and feature benchmark workflows
- local and Docker deployment modes

## Key Capabilities

- Strict day-ahead electricity consumption forecasting
- Configurable feature sets: calendar, exact daily lags, cyclical time, weather, recent dynamics
- PyTorch MLP training with early stopping and configurable hidden layers
- Reproducible dataset selection through `configs/common/data_sources.yaml`
- Model registry pattern via `artifacts/models/consumption/current`
- Historical replay with metrics and skipped-day tracking
- Benchmarking of multiple trained runs on the same replay window
- Benchmarking of multiple feature configurations end to end
- FastAPI routes for training, promotion, forecast, replay, and benchmarking
- Asynchronous API jobs for long-running operations
- Docker setup with one reusable image for CLI, API, and notebooks

## How The Forecasting Pipeline Works

The main consumption pipeline follows this logic.

### 1. Data sources are resolved from a catalog

The single source of truth for dataset paths is:

```bash
configs/common/data_sources.yaml
```

You can use dataset keys such as:

- `full_2020_2026`
- `clean_v1`
- `legacy_2020_2025`

Each dataset entry can define:

- historical consumption CSV
- benchmark or legacy forecast CSV
- weather CSV
- holidays workbook
- optional aliases for backward compatibility

### 2. Historical data is loaded and aligned

The training and inference runtime merges:

- historical consumption
- weather history
- holidays and special dates

This allows the feature builder to produce a consistent feature space both during training and during inference.

### 3. Features are generated with time-aware safeguards

The project supports feature families such as:

- calendar features
- exact day lags like `lag_d7`, `lag_d1`, `lag_d2`, etc.
- cyclical time features
- lag aggregates
- optional weather features
- optional recent intraday dynamics
- optional shifted recent dynamics

The operational forecast mode is `strict_day_ahead`. In that mode, the system prevents same-day target leakage. Feature sets that require same-day target observations are rejected for strict day-ahead usage.

### 4. The dataset is split by time

Training, validation, and test windows are controlled from YAML configs. The baseline config, for example, uses explicit end dates for train and validation windows instead of random shuffling.

### 5. Features and targets are scaled

The pipeline applies `MinMaxScaler` to both inputs and targets before training and stores the fitted scalers inside the run bundle so that inference can reuse them exactly.

### 6. A PyTorch MLP is trained

The model is a configurable multilayer perceptron implemented in `src/smartgrid/models/mlp.py`.

The trainer supports:

- configurable hidden layers
- dropout
- learning rate and weight decay
- early stopping via patience
- CPU or CUDA execution
- runtime profiling and batching diagnostics

### 7. Artifacts are persisted

Each training run stores:

- `model.pt`
- `x_scaler.pkl`
- `y_scaler.pkl`
- `run_summary.json`

Run outputs are written under:

- `artifacts/runs/consumption/<run_id>/`
- `artifacts/exports/consumption/<run_id>/`

### 8. A run can be promoted

Promotion copies a selected run bundle into:

```bash
artifacts/models/consumption/current
```

This directory is the production-style entry point used by forecast, replay, and API routes unless you explicitly override it.

### 9. Inference rebuilds the feature space from bundle metadata

At inference time, the system loads:

- the promoted model bundle
- the stored feature columns
- the training feature configuration
- the selected dataset configuration

It then rebuilds the target-day features and produces a strict day-ahead forecast for either:

- the next available day after the latest timestamp in history
- a user-specified target day

### 10. Historical replay evaluates operational behavior

Replay runs the day-ahead engine one day at a time over a historical interval. It only evaluates days with complete ground truth coverage and records skipped days when truth coverage is incomplete or when the requested bundle cannot generate valid features.

### 11. Benchmark workflows compare models and configs

The repository includes two benchmark layers:

- replay benchmark across several trained runs
- feature benchmark that trains several configs and ranks them on replay metrics

These workflows make it possible to compare not only offline validation metrics, but also stricter replay-time behavior.

## Repository Contents

Main directories:

- `src/smartgrid/`: application code
- `src/smartgrid/data/`: catalog resolution, loading, splits, and timeline logic
- `src/smartgrid/features/`: feature engineering
- `src/smartgrid/models/`: PyTorch MLP model
- `src/smartgrid/training/`: training and artifact persistence
- `src/smartgrid/inference/`: strict day-ahead inference and replay
- `src/smartgrid/registry/`: model bundle loading and ranking
- `src/smartgrid/cli/`: package CLI entry points
- `src/smartgrid/api/`: FastAPI application, API services, schemas, and async jobs
- `scripts/`: direct script wrappers and orchestration utilities
- `configs/`: datasets and experiment YAMLs
- `tests/`: automated checks
- `notebooks/`: demo and research notebooks
- `docs/`: supporting deployment and audit notes
- `data/processed/`: versioned processed datasets used by the demo

## Technology Stack

- Python 3.12
- PyTorch
- pandas and NumPy
- scikit-learn
- FastAPI and Uvicorn
- JupyterLab
- Docker and Docker Compose
- `uv` for dependency management and reproducible execution

## Current Repository Model

This repository intentionally versions the processed demo datasets under `data/processed/`. That makes the project easier to clone and run on a new machine without a separate data bootstrap step.

Tracked in Git:

- source code
- scripts
- configs
- tests
- notebooks
- processed datasets
- dependency metadata

Generated locally and not meant for Git:

- `artifacts/`
- virtual environments and caches
- editor-specific files
- optional local raw or interim data additions

## Installation

### Prerequisites

Local usage requires:

- Python `3.12`
- `uv`
- `make`

Check your environment:

```bash
python --version
uv --version
make --version
```

### Fresh Local Setup

```bash
git clone <your-repository-url>
cd smart-grid
git checkout dev
git pull origin dev
uv sync --all-groups
make verify
```

If `make verify` passes, the repository is ready for local development and demo usage.

### Dependency Profiles

Useful setup commands:

```bash
make install
make install-core
make install-dev
make install-dev-legacy
```

Use the `legacy` group only if you need the old TensorFlow or Keras-related extras.

## Data Setup And Validation

The main operational dataset key is:

```bash
full_2020_2026
```

The default convenience commands use these files:

- `data/processed/conso/Consumption data 2020-2026.csv`
- `data/processed/conso/Consumption forecast 2020-2026.csv`
- `data/processed/Weather data 2020-2026.csv`
- `data/processed/Holidays.xlsx`

Validate the local dataset state with:

```bash
make doctor
```

`make doctor` checks the catalog-resolved dataset files and reports missing paths before you start training or inference.

## Configuration Model

The project is driven by YAML configurations under `configs/consumption/`.

The main demo and operational baseline is:

```bash
configs/consumption/mlp_strict_day_ahead_baseline.yaml
```

That config defines:

- the dataset key
- the train/validation split boundaries
- the feature set
- the forecast mode
- the training hyperparameters
- the artifact output structure

Other configs extend the baseline with richer feature sets, including weather and shifted dynamics.

## Quick Start

If you just want to run the full workflow once on your local machine:

```bash
make doctor
make verify
make train-promote CONFIG=configs/consumption/mlp_strict_day_ahead_baseline.yaml ANALYSIS_DAYS=1 DATASET_KEY=full_2020_2026
make predict-next-day TARGET_DATE=2026-01-15
make replay-period START_DATE=2026-01-15 END_DATE=2026-01-15
make serve-api
```

Then open:

- `http://localhost:8000/docs`
- `http://localhost:8000/health`

## Daily Workflow With Make

The `Makefile` is the recommended operator entry point for day-to-day usage.

List available commands:

```bash
make
```

Most important targets:

```bash
make install
make doctor
make lint
make test
make build
make verify
make notebook
make serve-api
make train-consumption
make train-promote
make promote-consumption RUN_ID=consumption_mlp_...
make predict-next-day TARGET_DATE=2026-01-15
make replay-period START_DATE=2026-01-01 END_DATE=2026-01-31
make benchmark-features
make benchmark-replay MODEL_REFS="consumption_mlp_run_a consumption_mlp_run_b"
```

These targets use `uv run`, so you do not need to activate a virtual environment manually.

## Core Usage Scenarios

### Train a model

```bash
make train-consumption
```

Train a different feature configuration:

```bash
make train-consumption CONFIG=configs/consumption/mlp_strict_day_ahead_cyclical_weather_basic.yaml
```

### Train and promote immediately

```bash
make train-promote CONFIG=configs/consumption/mlp_strict_day_ahead_cyclical_weather_shifted_dynamics.yaml ANALYSIS_DAYS=3 DATASET_KEY=full_2020_2026
```

### Promote an existing run

```bash
make promote-consumption RUN_ID=consumption_mlp_20260415T205933Z
```

### Forecast a target day

```bash
make predict-next-day TARGET_DATE=2026-01-15
```

### Replay a historical period

```bash
make replay-period START_DATE=2026-01-01 END_DATE=2026-01-31
```

### Benchmark feature configurations

```bash
make benchmark-features
```

Override the compared configs if needed:

```bash
make benchmark-features \
  BENCHMARK_CONFIGS="configs/consumption/mlp_strict_day_ahead_baseline.yaml configs/consumption/mlp_strict_day_ahead_cyclical_weather_shifted_dynamics.yaml"
```

### Benchmark several trained runs on the same replay window

```bash
make benchmark-replay \
  START_DATE=2026-01-01 \
  END_DATE=2026-01-31 \
  MODEL_REFS="consumption_mlp_20260410T062646Z consumption_mlp_20260415T205933Z"
```

## Using The CLI Directly

If you prefer explicit commands instead of `make`, the repository keeps the script wrappers under `scripts/`.

Example commands:

```bash
uv run python scripts/train_consumption.py \
  --config configs/consumption/mlp_strict_day_ahead_baseline.yaml \
  --analysis-days 1 \
  --dataset-key full_2020_2026

uv run python scripts/promote_consumption_run.py \
  --run-id consumption_mlp_20260415T205933Z

uv run python scripts/predict_next_day.py \
  --dataset-key full_2020_2026 \
  --historical-csv "data/processed/conso/Consumption data 2020-2026.csv" \
  --weather-csv "data/processed/Weather data 2020-2026.csv" \
  --holidays-xlsx "data/processed/Holidays.xlsx" \
  --target-date 2026-01-15

uv run python scripts/replay_period.py \
  --dataset-key full_2020_2026 \
  --historical-csv "data/processed/conso/Consumption data 2020-2026.csv" \
  --weather-csv "data/processed/Weather data 2020-2026.csv" \
  --holidays-xlsx "data/processed/Holidays.xlsx" \
  --start-date 2026-01-01 \
  --end-date 2026-01-31
```

The package also exposes console entry points:

- `smartgrid-train-consumption`
- `smartgrid-promote-consumption`
- `smartgrid-predict-next-day`
- `smartgrid-replay-period`

## API

The FastAPI application adds an operational interface on top of the forecasting system. It is suitable for demos, integrations, and AI-agent-driven orchestration.
It is additive to the existing CLI and Make workflows, not a replacement for them.

The default promoted model directory is:

```bash
artifacts/models/consumption/current
```

Start the API locally:

```bash
make serve-api
```

Interactive documentation will be available at:

- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

### API Capabilities

Information and registry routes:

- `GET /`
- `GET /health`
- `GET /consumption/model-info`
- `GET /consumption/models`

Execution routes:

- `POST /consumption/train`
- `POST /consumption/promote`
- `POST /consumption/forecast/next-day`
- `POST /consumption/forecast/by-date`
- `POST /consumption/replay`
- `POST /consumption/benchmark/replay`
- `POST /consumption/benchmark/features`
- `POST /consumption/predict-from-features`

Async job routes:

- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`
- `POST /consumption/train/async`
- `POST /consumption/promote/async`
- `POST /consumption/replay/async`
- `POST /consumption/benchmark/replay/async`
- `POST /consumption/benchmark/features/async`

### What The API Returns

The API is designed to return business-useful outputs, not just raw predictions. Depending on the route, responses can include:

- run identifiers
- output CSV paths
- archive CSV paths
- replay metrics JSON paths
- per-day replay directories
- overall replay metrics
- skipped-day details
- fallback usage information

### API Examples

Train and promote a model:

```bash
curl -X POST http://localhost:8000/consumption/train \
  -H "Content-Type: application/json" \
  -d '{
    "config": "configs/consumption/mlp_strict_day_ahead_baseline.yaml",
    "analysis_days": 1,
    "dataset_key": "full_2020_2026",
    "promote": true
  }'
```

Forecast a specific day:

```bash
curl -X POST http://localhost:8000/consumption/forecast/by-date \
  -H "Content-Type: application/json" \
  -d '{
    "target_date": "2026-01-15",
    "dataset_key": "full_2020_2026",
    "write_outputs": true
  }'
```

Replay asynchronously:

```bash
curl -X POST http://localhost:8000/consumption/replay/async \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2026-01-01",
    "end_date": "2026-01-31",
    "dataset_key": "full_2020_2026",
    "write_outputs": true
  }'
```

Poll the job result:

```bash
curl http://localhost:8000/jobs/<job_id>/result
```

### Important API Note

Background API jobs are currently process-local and in-memory. That is perfectly adequate for demo and operator workflows, but it is not a durable distributed job queue.

## Docker Deployment

The Docker setup is built for daily development, demos, and near-production testing.

### Docker Design

The Docker stack intentionally uses one reusable development image:

- `cli` builds the image
- `api` reuses the same image
- `notebook` reuses the same image

The repository is mounted into the container at runtime with:

```bash
.:/workspace
```

This has important consequences:

- your code changes are immediately visible inside containers
- generated artifacts stay persistent on the host
- notebooks, configs, and scripts remain editable without rebuilding the image
- the `uv` cache is preserved in a named Docker volume for faster restarts

### Files In The Docker Stack

- `Dockerfile`
- `docker-compose.yml`
- `docker-compose.gpu.yml`
- `docker/entrypoint.sh`

### What The Docker Entrypoint Does

On startup, the entrypoint:

- switches to `/workspace`
- ensures runtime directories exist
- synchronizes the mounted repo with `uv sync`
- runs `make doctor` by default
- optionally runs `make verify`

### CPU Quick Start

```bash
docker compose build cli --progress=plain
docker compose run --rm cli
docker compose up api
```

### GPU Quick Start

Use the base file plus the GPU override only on NVIDIA-capable machines:

```bash
docker compose build cli --progress=plain
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm cli
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up api
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile notebook up notebook
```

### API And Notebook Services

Start the API:

```bash
docker compose up api
```

Start JupyterLab:

```bash
docker compose --profile notebook up notebook
```

### Persistence In Docker

Because the full repository is mounted into `/workspace`, the following remain persistent between runs:

- `artifacts/`
- `data/raw/`
- `data/interim/`
- `data/external/`
- `data/processed/`
- local code and notebooks

## Notebooks

Launch JupyterLab locally:

```bash
make notebook
```

Useful notebooks include:

- `notebooks/experiments/SmartGrid_CLI_Demo_Notebook_v4.ipynb`
- `notebooks/experiments/SmartGrid_Demo_Globale.ipynb`
- `notebooks/experiments/SmartGrid_Benchmark_Global.ipynb`

The notebooks are intended as demo, validation, and exploratory layers on top of the same underlying CLI and pipeline.

## Artifacts And Outputs

Generated outputs are written under `artifacts/`.

Important directories:

- `artifacts/runs/consumption/`
- `artifacts/exports/consumption/`
- `artifacts/models/consumption/current/`
- `artifacts/forecasts/consumption/`
- `artifacts/replays/consumption/`
- `artifacts/benchmarks/`
- `artifacts/logs/`

Typical contents include:

- promoted bundles
- per-run model files and scalers
- training summaries
- notebook compatibility exports
- forecast CSVs
- replay CSVs and replay metrics JSON files
- benchmark summaries and manifests
- log files for train, predict, and replay workflows

## Validation And Development Commands

Quality and packaging checks:

```bash
make lint
make test
make build
make verify
```

The test suite includes unit coverage for:

- feature engineering
- timeline semantics
- metrics and reporting
- data catalog behavior
- FastAPI smoke routes

## Recommended End-To-End Demo Flow

For a clean demo on a local machine:

```bash
make doctor
make verify
make train-promote CONFIG=configs/consumption/mlp_strict_day_ahead_baseline.yaml ANALYSIS_DAYS=1 DATASET_KEY=full_2020_2026
make predict-next-day TARGET_DATE=2026-01-15
make replay-period START_DATE=2026-01-15 END_DATE=2026-01-15
make serve-api
```

Then, in another terminal:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/consumption/forecast/by-date \
  -H "Content-Type: application/json" \
  -d '{
    "target_date": "2026-01-15",
    "dataset_key": "full_2020_2026",
    "write_outputs": true
  }'
```

## Operational Notes

- The Makefile is the safest and most stable entry point for normal usage.
- Forecast and replay expect a promoted model in `artifacts/models/consumption/current` unless you override the bundle directory.
- If you want GPU execution in Docker, use the GPU Compose override only on machines with an NVIDIA runtime.
- If a route or command fails because of missing data files, run `make doctor` first and confirm the resolved catalog paths exist.

## License And Ownership

This project was created by Khalid Bensaber. Unless a license file is provided, no open-source license is granted.