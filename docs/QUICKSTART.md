# Quick Start

Audience: supervisors, evaluators, and new teammates who want the cleanest success path first.

This guide covers the baseline local and Docker path. It does not explain architecture internals or advanced customization.

## Prerequisites

- Python `3.12`
- `uv`
- `make`
- Git
- Optional: Docker and Docker Compose
- Optional: an NVIDIA GPU if you want CUDA inside Docker

Quick checks:

```bash
python --version
uv --version
make --version
docker --version
```

## Clone And Checkout

```bash
git clone <repository-url> SmartGrid
cd SmartGrid
git checkout dev
```

If you already have the repository locally, update your branch before using the commands below.

## Install

Install all dependency groups:

```bash
make install
```

Equivalent direct command:

```bash
uv sync --all-groups
```

Expected result: the virtual environment is populated and `uv run ...` commands work without extra setup.

## Run `make doctor`

```bash
make doctor
```

What it checks:

- the dataset keys exposed by `configs/common/data_sources.yaml`
- the tracked CSV and XLSX files referenced by the catalog

Expected result: each configured path is marked `OK` and the command ends with `Setup looks complete.`

## Run `make verify`

```bash
make verify
```

What it runs:

- `make doctor`
- `make test`
- `make build`

Expected result: the repository passes dataset checks, the test suite, and the package build.

## Train And Promote A Baseline Model

The shortest happy path is:

```bash
make train-promote
```

If you want to be explicit about the defaults:

```bash
make train-promote \
  CONFIG=configs/consumption/mlp_strict_day_ahead_baseline.yaml \
  ANALYSIS_DAYS=1 \
  DATASET_KEY=full_2020_2026
```

Expected outputs:

- a new run under `artifacts/runs/consumption/<run_id>/`
- export files under `artifacts/exports/consumption/<run_id>/`
- a promoted bundle under `artifacts/models/consumption/current/`
- a train log under `artifacts/logs/train/`

## Forecast One Day

```bash
make predict-next-day TARGET_DATE=2026-01-15
```

Expected outputs:

- `artifacts/forecasts/consumption/current/forecast_2026-01-15.csv`
- `artifacts/forecasts/consumption/archive/<run_id>/forecast_2026-01-15.csv`
- a predict log under `artifacts/logs/predict/`

Important: prediction loads the promoted bundle from `artifacts/models/consumption/current`.

## Replay A Historical Period

```bash
make replay-period START_DATE=2026-01-01 END_DATE=2026-01-31
```

Expected outputs:

- `artifacts/replays/consumption/<stamp>__2026-01-01__2026-01-31/replay_forecasts.csv`
- `artifacts/replays/consumption/<stamp>__2026-01-01__2026-01-31/replay_metrics.json`
- a replay log under `artifacts/logs/replay/`

Replay is the official operational evaluation path for the repository.

## Start The API

```bash
make serve-api
```

Then open:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

Expected result: the API can inspect the current model and serve forecast, replay, training, promotion, and benchmark routes.

## Docker Quick Start

Build the shared development image:

```bash
docker compose build cli --progress=plain
```

Open a CLI container:

```bash
docker compose run --rm cli
```

Start the API:

```bash
docker compose up api
```

GPU machine:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm cli
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up api
```

Docker notes:

- `cli`, `api`, and `notebook` reuse the same image
- the repository is bind-mounted into `/workspace`
- the entrypoint runs `uv sync` and `make doctor` automatically
- tracked demo data under `data/processed/` stays visible inside the container

## What Success Looks Like

- `make doctor` confirms the catalog files exist
- `make verify` passes
- `make train-promote` writes a new run and a current promoted bundle
- `make predict-next-day` writes one forecast CSV for the requested day
- `make replay-period` writes replay forecasts plus metrics JSON
- `make serve-api` exposes `/docs` without extra manual wiring

## Read Next

- [Operations and Deployment](OPERATIONS_AND_DEPLOYMENT.md)
- [Architecture and Code Map](ARCHITECTURE_AND_CODE_MAP.md)
- [Notebook and Demo Guide](NOTEBOOK_AND_DEMO_GUIDE.md)
