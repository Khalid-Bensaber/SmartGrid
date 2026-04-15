# Smart Grid

Electricity consumption forecasting pipeline centered on `src/smartgrid/`.

## Overview

This project currently focuses mainly on the consumption workflow:
- training MLP models with PyTorch
- promoting a run to `current`
- forecasting day+1 (next-day prediction)
- replaying historical periods over a date range
- benchmarking multiple configurations or runs
- analysis notebooks in `notebooks/experiments/`

## Environment

Required Python version: `>= 3.12`

Recommended installation with `uv`:

```bash
uv sync --all-groups
```

Useful variants:

```bash
uv sync
uv sync --group dev
uv sync --all-groups
```

Available `make` shortcuts:

```bash
make install
make install-core
make install-dev
make install-dev-legacy
make lint
make test
make train-consumption
make serve-api
```

## Data and protocol

The single source of truth for datasets is:

```bash
configs/common/data_sources.yaml
```

The consumption configs (`configs/consumption/*.yaml`) are currently aligned to:
- `data.dataset_key: full_2020_2026`
- `train_end_date: '2025-09-30'`
- `val_end_date: '2025-12-31'`

So today:
- train: up to `2025-09-30`
- validation: from `2025-10-01` to `2025-12-31`
- test: only `2026`

Files used by `full_2020_2026`:
- historical: `data/processed/conso/Consumption data 2020-2026.csv`
- legacy forecasts: `data/processed/conso/Consumption forecast 2020-2026.csv`
- weather: `data/processed/Weather data 2020-2026.csv`
- holidays: `data/processed/Holidays.xlsx`

## Main commands

The wrappers in `scripts/` are the simplest commands to run locally.
There are also installed entry points (`smartgrid-*`), but this README uses the wrappers to stay explicit.

### 1. Train a model

Baseline example:

```bash
uv run python scripts/train_consumption.py \
  --config configs/consumption/mlp_baseline.yaml
```

Example with immediate promotion:

```bash
uv run python scripts/train_consumption.py \
  --config configs/consumption/mlp_weather_basic.yaml \
  --promote
```

Example forcing the analysis day in the run exports:

```bash
uv run python scripts/train_consumption.py \
  --config configs/consumption/mlp_weather_all.yaml \
  --analysis-date 2026-01-15 \
  --analysis-days 3
```

Example overriding dataset or file paths:

```bash
uv run python scripts/train_consumption.py \
  --config configs/consumption/mlp_baseline.yaml \
  --dataset-key full_2020_2026 \
  --historical-csv "data/processed/conso/Consumption data 2020-2026.csv" \
  --benchmark-csv "data/processed/conso/Consumption forecast 2020-2026.csv" \
  --weather-csv "data/processed/Weather data 2020-2026.csv" \
  --holidays-xlsx "data/processed/Holidays.xlsx"
```

Main outputs of a training run:
- `artifacts/runs/consumption/<RUN_ID>/`
- `artifacts/exports/consumption/<RUN_ID>/backtest.csv`
- `artifacts/exports/consumption/<RUN_ID>/selected_day_<DATE>.csv`
- `artifacts/models/consumption/current/` if `--promote` is used

### 2. Promote an existing run

```bash
uv run python scripts/promote_consumption_run.py \
  --run-id consumption_mlp_20260415T205933Z
```

### 3. Predict the full next day (day+1)

Using the currently promoted model:

```bash
uv run python scripts/predict_next_day.py \
  --historical-csv "data/processed/conso/Consumption data 2020-2026.csv" \
  --weather-csv "data/processed/Weather data 2020-2026.csv" \
  --holidays-xlsx "data/processed/Holidays.xlsx" \
  --target-date 2026-01-15 \
  --output-csv artifacts/forecasts/manual/forecast_2026-01-15.csv
```

Without automatic fallback to another compatible run:

```bash
uv run python scripts/predict_next_day.py \
  --historical-csv "data/processed/conso/Consumption data 2020-2026.csv" \
  --weather-csv "data/processed/Weather data 2020-2026.csv" \
  --holidays-xlsx "data/processed/Holidays.xlsx" \
  --target-date 2026-01-15 \
  --disable-fallback
```

### 4. Replay a historical period with the current model

Simple replay:

```bash
uv run python scripts/replay_period.py \
  --historical-csv "data/processed/conso/Consumption data 2020-2026.csv" \
  --weather-csv "data/processed/Weather data 2020-2026.csv" \
  --holidays-xlsx "data/processed/Holidays.xlsx" \
  --start-date 2026-01-01 \
  --end-date 2026-01-31
```

Replay with one CSV exported per day:

```bash
uv run python scripts/replay_period.py \
  --historical-csv "data/processed/conso/Consumption data 2020-2026.csv" \
  --weather-csv "data/processed/Weather data 2020-2026.csv" \
  --holidays-xlsx "data/processed/Holidays.xlsx" \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --write-per-day
```

Main outputs:
- `artifacts/replays/consumption/<STAMP>__<START>__<END>/replay_forecasts.csv`
- `artifacts/replays/consumption/<STAMP>__<START>__<END>/replay_metrics.json`

### 5. Benchmark multiple training configs

Compare several YAML configs under the same protocol:

```bash
uv run python scripts/benchmark_feature_variants.py \
  configs/consumption/mlp_baseline.yaml \
  configs/consumption/mlp_weather_basic.yaml \
  configs/consumption/mlp_weather_all.yaml \
  --output-csv artifacts/benchmarks/consumption_feature_variants.csv \
  --analysis-days 1
```

The benchmark CSV aggregates fields such as:
- `run_id`
- `selected_analysis_day`
- `MAE`
- `RMSE`
- split sizes
- feature information

### 6. Replay-benchmark multiple already-trained runs

Compare multiple runs over the same date range without retraining:

```bash
uv run python scripts/benchmark_replay_models.py \
  --historical-csv "data/processed/conso/Consumption data 2020-2026.csv" \
  --weather-csv "data/processed/Weather data 2020-2026.csv" \
  --holidays-xlsx "data/processed/Holidays.xlsx" \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  consumption_mlp_20260410T062646Z \
  consumption_mlp_20260414T210959Z \
  consumption_mlp_20260415T205933Z
```

With fallback enabled:

```bash
uv run python scripts/benchmark_replay_models.py \
  --historical-csv "data/processed/conso/Consumption data 2020-2026.csv" \
  --weather-csv "data/processed/Weather data 2020-2026.csv" \
  --holidays-xlsx "data/processed/Holidays.xlsx" \
  --start-date 2026-01-01 \
  --end-date 2026-01-31 \
  --allow-fallback \
  consumption_mlp_20260415T205933Z
```

Main outputs:
- `artifacts/benchmarks/replay/<STAMP>__<START>__<END>/replay_benchmark_summary.csv`
- `artifacts/benchmarks/replay/<STAMP>__<START>__<END>/replay_benchmark_manifest.json`
- one subfolder per benchmarked run

## Demo notebook

Main notebook:

```bash
notebooks/experiments/SmartGrid_Demo_Globale.ipynb
```

Start:

```bash
uv run jupyter lab
```

Important notebook variables:
- `DATASET_KEY`: dataset in use
- `RUN_TRAINING_BENCHMARK`: whether to train models
- `RUN_REPLAY_BENCHMARK`: whether to run multi-day replay
- `REPLAY_REQUIRE_FRESH_RUNS`: if `False`, reuse existing runs
- `DETAILED_ANALYSIS_DATE`: day (day+1) to recompute for detailed analysis
- `REPLAY_START_DATE` / `REPLAY_END_DATE`: replay range

Recommended usage:
- train models once
- then reuse existing runs
- run replays or analyses on other dates without retraining

## API

The FastAPI server loads the promoted model from:

```bash
artifacts/models/consumption/current
```

Run the API:

```bash
uv run uvicorn smartgrid.api:app --reload --host 0.0.0.0 --port 8000
```

Main endpoints:
- `GET /`
- `GET /health`
- `GET /consumption/model-info`
- `POST /consumption/predict-from-features`

## Quality and verification

Lint:

```bash
uv run ruff check src tests scripts
```

Tests:

```bash
uv run pytest
```

Quick package compilation:

```bash
uv run python -m compileall src
```

## Useful tree

- `configs/common/data_sources.yaml`: central dataset catalogue
- `configs/consumption/`: consumption model configs
- `scripts/`: local CLI wrappers
- `src/smartgrid/data/`: loading and temporal splits
- `src/smartgrid/features/`: feature engineering
- `src/smartgrid/training/`: training and artifact management
- `src/smartgrid/inference/`: prediction and replay
- `src/smartgrid/registry/`: promotion and bundle loading
- `src/smartgrid/api/`: FastAPI server
- `artifacts/`: execution outputs
- `notebooks/experiments/`: analysis notebooks
