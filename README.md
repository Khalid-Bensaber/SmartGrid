# SmartGrid

SmartGrid is an end-to-end electricity consumption forecasting repository built around reproducible training, strict day-ahead inference, historical replay, notebooks, a FastAPI layer, and Docker-based demo workflows.

## Project Overview

The repository works on processed `10min` electricity-consumption data covering multiple buildings, plus aligned weather and holiday inputs. Its main operational target is the aggregated consumption signal `tot`, reconstructed from the tracked building-level consumption columns.

## Problem Statement

The real-world problem is day-ahead electricity forecasting: produce tomorrow's full load profile before tomorrow starts, using only information that would genuinely be available at that time. That operational constraint matters because the project must not rely on same-day target observations or optimistic offline shortcuts.

## Current Implemented Solution

The current implemented solution is a strict day-ahead consumption forecasting pipeline built around:

- dataset resolution through `configs/common/data_sources.yaml`
- time-aware feature engineering
- PyTorch MLP training and bundle promotion
- forecast-time feature rebuilding from the promoted bundle
- historical replay as the official operational evaluation path

The current default training config is `configs/consumption/mlp_strict_day_ahead_cyclical_weather_shifted_dynamics_h512_256_128_do010_wd1em5.yaml`. It trains a strict day-ahead MLP on the `full_2020_2026` dataset with calendar features, temperature, manual daily lags, cyclical time, lag aggregates, shifted recent dynamics, and `basic` weather features, while explicitly disabling same-day recent dynamics.

## Repository Scope

The production-style scope implemented and documented today is electricity consumption forecasting. The repository also contains photovoltaic data assets and extension points, but the maintained CLI, API, replay, and handoff documentation are currently centered on the consumption pipeline.

The repo is intended to be used in several aligned ways:

- as a local `make`-driven project
- as a CLI toolkit through `scripts/`
- as an HTTP orchestration surface through FastAPI
- as a notebook-backed demo and analysis environment
- as a Dockerized handoff environment for supervisors and maintainers

## What This Repository Does

- Trains consumption models through the main orchestration path in `scripts/train_consumption.py`
- Promotes the selected bundle to `artifacts/models/consumption/current`
- Forecasts the next or chosen day with strict day-ahead semantics
- Replays historical periods to evaluate operational behavior day by day
- Exposes training, promotion, forecast, replay, and benchmarking through FastAPI
- Supports local `make` workflows, direct CLI usage, notebooks, and Docker

## Quick Start

```bash
git clone <repository-url> SmartGrid
cd SmartGrid
git checkout dev
make install
make doctor
make verify
make train-promote
make predict-next-day
make replay-period START_DATE=2026-01-01 END_DATE=2026-01-31
make serve-api
```

After `make serve-api`, open `http://localhost:8000/docs`.

## Most Important Make Commands

- `make install`: install all dependency groups with `uv`
- `make doctor`: verify the tracked dataset files referenced by the catalog
- `make verify`: run setup checks, tests, and a package build
- `make train-promote`: train and immediately promote the new run
- `make promote-consumption RUN_ID=...`: promote an existing run manually
- `make predict-next-day`: forecast the next available day after the latest history timestamp, or pass `TARGET_DATE=YYYY-MM-DD` to force a specific day
- `make replay-period START_DATE=... END_DATE=...`: run official historical replay
- `make serve-api`: start the FastAPI server on `0.0.0.0:8000`
- `make benchmark-features`: train several configs and rank them on replay
- `make benchmark-replay MODEL_REFS='run_a run_b'`: compare existing runs on one replay window

## Operational Notes

- `strict_day_ahead` is the explicit operational forecasting mode. Same-day target leakage must not be reintroduced.
- Historical `replay` is the official operational evaluation path. Offline train/test metrics remain useful, but they are secondary diagnostics.
- The promoted bundle under `artifacts/models/consumption/current` is the default runtime entry point for CLI and API forecasting.
- `make predict-next-day` is automatic by default: if `TARGET_DATE` is omitted, the runtime forecasts the next available day after the latest timestamp in history.
- A genuine future-date forecast can return `Ptot_TOTAL_Real = null` because no ground truth exists yet. Immediate runtime metrics are therefore unavailable for true future days; use historical replay or evaluate later when truth arrives.
- Processed demo datasets are versioned under `data/processed/`; generated outputs and logs go under `artifacts/`.
- API async jobs are in-memory and process-local. They are suitable for demos and light orchestration, not durable distributed scheduling.

## Repository At A Glance

- `configs/`: dataset catalog and experiment YAML files
- `scripts/`: operator-facing entry scripts
- `src/smartgrid/`: application code for data, features, training, inference, registry, API, and notebook helpers
- `tests/`: regression coverage for temporal semantics, features, catalog resolution, and API contracts
- `notebooks/`: demo and research notebooks
- `docs/`: handoff, operations, architecture, and integration documentation

## Historical Material

Superseded notes, older deployment writeups, and notebook audit documents are kept under `docs/archive/` when they are still useful for project history.

## Documentation Index

- [README.md](README.md)
- [docs/QUICKSTART.md](docs/QUICKSTART.md)
- [docs/OPERATIONS_AND_DEPLOYMENT.md](docs/OPERATIONS_AND_DEPLOYMENT.md)
- [docs/API_AND_SCHEDULER_INTEGRATION.md](docs/API_AND_SCHEDULER_INTEGRATION.md)
- [docs/ARCHITECTURE_AND_CODE_MAP.md](docs/ARCHITECTURE_AND_CODE_MAP.md)
- [docs/CUSTOMIZATION_GUIDE.md](docs/CUSTOMIZATION_GUIDE.md)
- [docs/DATA_BACKEND_MIGRATION.md](docs/DATA_BACKEND_MIGRATION.md)
- [docs/NOTEBOOK_AND_DEMO_GUIDE.md](docs/NOTEBOOK_AND_DEMO_GUIDE.md)
- [MAINTAINER_GUIDE.md](MAINTAINER_GUIDE.md)
