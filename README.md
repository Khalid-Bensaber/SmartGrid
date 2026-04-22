# SmartGrid

SmartGrid is an end-to-end electricity consumption forecasting repository. It covers dataset resolution, feature engineering, PyTorch training, model promotion, strict day-ahead inference, historical replay, notebooks, a FastAPI layer, and Docker-based demo/deployment workflows.

The current operational surface is the consumption pipeline. The repository also contains photovoltaic data assets, but production-style CLI and API workflows are currently documented and implemented for consumption forecasting.

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
make predict-next-day TARGET_DATE=2026-01-15
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
- `make predict-next-day TARGET_DATE=YYYY-MM-DD`: forecast one target day from the promoted bundle
- `make replay-period START_DATE=... END_DATE=...`: run official historical replay
- `make serve-api`: start the FastAPI server on `0.0.0.0:8000`
- `make benchmark-features`: train several configs and rank them on replay
- `make benchmark-replay MODEL_REFS='run_a run_b'`: compare existing runs on one replay window

## Documentation Map

- [Quick Start](docs/QUICKSTART.md)
- [Operations and Deployment](docs/OPERATIONS_AND_DEPLOYMENT.md)
- [API and Scheduler Integration](docs/API_AND_SCHEDULER_INTEGRATION.md)
- [Architecture and Code Map](docs/ARCHITECTURE_AND_CODE_MAP.md)
- [Customization Guide](docs/CUSTOMIZATION_GUIDE.md)
- [Data Backend Migration](docs/DATA_BACKEND_MIGRATION.md)
- [Notebook and Demo Guide](docs/NOTEBOOK_AND_DEMO_GUIDE.md)
- [Maintainer Guide](MAINTAINER_GUIDE.md)

## Operational Notes

- `strict_day_ahead` is the explicit operational forecasting mode. Same-day target leakage must not be reintroduced.
- Historical `replay` is the official operational evaluation path. Offline train/test metrics remain useful, but they are secondary diagnostics.
- The promoted bundle under `artifacts/models/consumption/current` is the default runtime entry point for CLI and API forecasting.
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
