# Smart Grid

Smart Grid is a Python project for electricity consumption forecasting. It includes data loading, feature engineering, PyTorch training, run promotion, day-ahead inference, replay benchmarking, notebooks, and a small FastAPI service.

## Repository Status

This repository is currently public, and the processed datasets under `data/processed/` are intentionally versioned so the project can be cloned and used on a new machine without a separate data bootstrap step.

What is tracked in Git:
- application code in `src/`
- CLI wrappers in `scripts/`
- configuration in `configs/`
- tests in `tests/`
- notebooks in `notebooks/`
- processed datasets in `data/processed/`
- reproducible dependency metadata in `pyproject.toml` and `uv.lock`

What stays local:
- generated outputs in `artifacts/`
- caches, virtual environments, and editor files
- raw, interim, and external data folders unless you add files there on purpose

## Requirements

- Python `3.12`
- `uv`

Check your versions:

```bash
python --version
uv --version
```

## Fresh Setup On A New PC

Once your latest changes are pushed, a new machine should only need:

```bash
git clone <your-repo-url>
cd smart-grid
git checkout dev
git pull origin dev
uv sync --all-groups
make verify
```

If `make verify` passes, the repository is ready for day-to-day work on that machine.

## Daily Workflow With Make

The `Makefile` is meant to be the main entry point for common tasks. Run this to see the available commands:

```bash
make
```

Most useful targets:

```bash
make install
make doctor
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

## Dataset Catalog

The single source of truth for data paths is:

```bash
configs/common/data_sources.yaml
```

The main consumption dataset keys currently available are:
- `full_2020_2026` (official strict day-ahead dataset)
- `clean_v1`
- `legacy_2020_2025`

The default day-to-day files used by the convenience commands in the `Makefile` are:
- `data/processed/conso/Consumption data 2020-2026.csv`
- `data/processed/conso/Consumption forecast 2020-2026.csv`
- `data/processed/Weather data 2020-2026.csv`
- `data/processed/Holidays.xlsx`

You can validate that the expected dataset files are present with:

```bash
make doctor
```

## Common Commands

### Train a model

Strict day-ahead baseline training:

```bash
make train-consumption
```

Train a different config:

```bash
make train-consumption CONFIG=configs/consumption/mlp_strict_day_ahead_cyclical_weather_basic.yaml
```

Train and promote immediately:

```bash
make train-promote CONFIG=configs/consumption/mlp_strict_day_ahead_cyclical_weather_shifted_dynamics.yaml ANALYSIS_DAYS=3
```

### Promote an existing run

```bash
make promote-consumption RUN_ID=consumption_mlp_20260415T205933Z
```

### Forecast the next day

```bash
make predict-next-day TARGET_DATE=2026-01-15
```

### Replay a historical period

```bash
make replay-period START_DATE=2026-01-01 END_DATE=2026-01-31
```

### Compare feature configurations

```bash
make benchmark-features
```

Override the compared configs if needed:

```bash
make benchmark-features \
  BENCHMARK_CONFIGS="configs/consumption/mlp_strict_day_ahead_baseline.yaml configs/consumption/mlp_strict_day_ahead_cyclical_weather_shifted_dynamics.yaml"
```

### Compare several trained runs on the same replay window

```bash
make benchmark-replay \
  START_DATE=2026-01-01 \
  END_DATE=2026-01-31 \
  MODEL_REFS="consumption_mlp_20260410T062646Z consumption_mlp_20260415T205933Z"
```

## Running Without Make

If you prefer direct commands, the wrappers in `scripts/` remain available:

```bash
uv run python scripts/train_consumption.py --config configs/consumption/mlp_baseline.yaml
uv run python scripts/promote_consumption_run.py --run-id consumption_mlp_20260415T205933Z
uv run python scripts/predict_next_day.py --historical-csv "data/processed/conso/Consumption data 2020-2026.csv" --weather-csv "data/processed/Weather data 2020-2026.csv" --holidays-xlsx "data/processed/Holidays.xlsx" --target-date 2026-01-15
uv run python scripts/replay_period.py --historical-csv "data/processed/conso/Consumption data 2020-2026.csv" --weather-csv "data/processed/Weather data 2020-2026.csv" --holidays-xlsx "data/processed/Holidays.xlsx" --start-date 2026-01-01 --end-date 2026-01-31
```

## API

The FastAPI app loads the promoted model bundle from:

```bash
artifacts/models/consumption/current
```

Start the API locally:

```bash
make serve-api
```

Main endpoints:
- `GET /`
- `GET /health`
- `GET /consumption/model-info`
- `POST /consumption/forecast/next-day`
- `POST /consumption/forecast/by-date`
- `POST /consumption/replay`
- `POST /consumption/predict-from-features`

The business-facing forecast and replay routes now use the strict day-ahead engine.
`POST /consumption/predict-from-features` remains available as a lower-level diagnostic endpoint.

## Notebooks

Launch JupyterLab with:

```bash
make notebook
```

Main notebook:
- `notebooks/experiments/SmartGrid_Demo_Globale.ipynb`

## Validation

Code quality and packaging checks:

```bash
make lint
make test
make build
make verify
```

## Output Directories

Generated outputs are written under `artifacts/`, especially:
- `artifacts/runs/consumption/`
- `artifacts/exports/consumption/`
- `artifacts/models/consumption/current/`
- `artifacts/replays/consumption/`
- `artifacts/benchmarks/`
- `artifacts/logs/`

These outputs are intentionally ignored by Git.

## Project Layout

- `src/smartgrid/`: core package
- `src/smartgrid/data/`: catalog resolution, loading, and temporal splits
- `src/smartgrid/features/`: feature engineering
- `src/smartgrid/training/`: model training and bundle persistence
- `src/smartgrid/inference/`: day-ahead inference and replay runtime
- `src/smartgrid/registry/`: run promotion and model bundle loading
- `src/smartgrid/api/`: FastAPI application
- `scripts/`: explicit wrapper scripts
- `configs/`: datasets and experiment configurations
- `tests/`: automated checks
- `data/processed/`: versioned processed datasets
- `artifacts/`: generated local outputs
