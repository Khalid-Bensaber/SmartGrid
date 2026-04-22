# Architecture and Code Map

Audience: maintainers, graders, and teammates who need to understand where the logic lives before changing it.

This guide explains the current consumption pipeline as it exists in code today.

## High-Level Flow

```text
historical consumption CSV + weather CSV + holidays workbook
        -> dataset catalog resolution
        -> dataframe loading and timestamp validation
        -> feature engineering
        -> chronological train/val/test split
        -> scaling
        -> PyTorch MLP training
        -> run bundle + export artifacts
        -> promotion to artifacts/models/consumption/current
        -> strict day-ahead forecast or replay
        -> optional API and notebook orchestration
```

## Root Directory Map

### `configs/`

Configuration layer.

- `configs/common/data_sources.yaml`: source of truth for dataset keys and file paths
- `configs/consumption/*.yaml`: current consumption experiment configurations
- `configs/production/`: reserved but not yet operationalized in the active pipeline

### `scripts/`

Thin operator-facing entry points.

- `train_consumption.py`: main training entry script
- `promote_consumption_run.py`: manual promotion
- `predict_next_day.py`: strict day-ahead forecast
- `replay_period.py`: official operational replay
- `benchmark_feature_variants.py`: train several configs, then rank them by replay
- `benchmark_replay_models.py`: compare existing run bundles on one replay window
- `check_setup.py`: dataset catalog validation used by `make doctor`
- `generate_cli_demo_notebook_v4.py`: rebuilds the current demo notebook

### `src/smartgrid/`

Main application package.

### `src/Legacy/`

Historical reference code only. It is not the path to extend for the current modular pipeline.

### `tests/`

Regression coverage for strict day-ahead semantics, feature parity, catalog resolution, replay behavior, and API contracts.

### `data/processed/`

Tracked demo and evaluation datasets. The active consumption pipeline reads from here through the dataset catalog. PV data assets also exist here, but they are not yet wired into a parallel operational pipeline.

### `artifacts/`

Generated outputs, logs, replays, forecasts, benchmarks, promoted bundles, and notebook exports.

### `docs/`

Handoff, operations, architecture, customization, migration, and integration guides.

## `src/smartgrid/` Module Map

### `src/smartgrid/common/`

Shared utilities and constants.

- `constants.py`: total-building columns, weather column groups, forecast cadence, and strict day-ahead constants
- `paths.py`: standard output paths for runs, forecasts, and replays
- `utils.py`: YAML loading, device selection, run id creation, and directory helpers
- `logging.py`: log file path creation and logger setup
- `profiling.py`: training/runtime profiling helpers

### `src/smartgrid/data/`

Dataset resolution and loading.

- `catalog.py`: resolves dataset keys from `configs/common/data_sources.yaml`
- `loaders.py`: loads history, weather, holidays, benchmark data, and target-day frames
- `splits.py`: ratio-based and date-based chronological splitting
- `timeline.py`: timestamp validation, gap detection, and day coverage checks

Important current behavior:

- `load_history()` reconstructs the `tot` target from the four building columns defined in `TOTAL_COLUMNS`
- partial totals are rejected by using `min_count=len(TOTAL_COLUMNS)`
- duplicate timestamps raise an error instead of being silently accepted

### `src/smartgrid/features/`

Feature engineering and forecast-time feature reconstruction.

- `engineering.py`: calendar features, daily lags, lag aggregates, shifted recent dynamics, weather feature selection, validity filtering, and strict day-ahead safeguards

This module is the most important place to keep training and inference semantics aligned.

### `src/smartgrid/models/`

Model definitions.

- `mlp.py`: the current configurable dense PyTorch MLP used for consumption training

### `src/smartgrid/training/`

Training loop and artifact persistence.

- `trainer.py`: batching, optimization, early stopping, and prediction helpers
- `artifacts.py`: saves model/scaler bundles and handles promotion to `current`

### `src/smartgrid/inference/`

Runtime forecasting and replay.

- `consumption.py`: prediction helpers from feature rows or feature matrices
- `day_ahead.py`: bundle loading, target-day feature rebuilding, forecast execution, replay loop, and optional fallback bundle selection

Key semantics:

- only `strict_day_ahead` is supported in the official replay flow
- replay evaluates one full day at a time
- days with incomplete truth coverage are skipped explicitly and recorded

### `src/smartgrid/evaluation/`

Metrics and reporting outputs.

- `metrics.py`: MAE, RMSE, bias, tolerance-based metrics, and comparison helpers
- `reporting.py`: offline backtest construction, replay evaluation, and export helpers

### `src/smartgrid/registry/`

Bundle loading and bundle ranking.

- `model_registry.py`: loads a bundle from disk and ranks candidate bundles for fallback use

### `src/smartgrid/api/`

FastAPI orchestration layer.

- `app.py`: route registration
- `schemas.py`: request and response models
- `services.py`: training, promotion, forecast, replay, and benchmark service logic
- `jobs.py`: in-memory async job manager backed by a thread pool

### `src/smartgrid/notebooks/`

Notebook helper code.

- `cli_demo_utils.py`: helper functions used by the V4 demo notebook for orchestration and analysis

## Config Map

### `configs/common/data_sources.yaml`

Defines:

- dataset keys such as `full_2020_2026`, `clean_v1`, and `legacy_2020_2025`
- historical CSV paths
- benchmark or legacy forecast paths
- weather paths
- holidays workbook paths
- optional aliases for backward compatibility

### `configs/consumption/*.yaml`

Defines:

- `data`: dataset key, date column, target name
- `split`: train/validation/test boundaries
- `features`: strict day-ahead feature flags and lag days
- `training`: seed, batch size, optimizer parameters, hidden layers, device
- `artifacts`: where to write runs, exports, and registry bundles

## Data Flow In Practice

### Training Flow

1. `scripts/train_consumption.py` calls `src/smartgrid/cli/train_consumption.py`
2. The dataset catalog resolves the current file paths
3. History, weather, and holidays are loaded
4. `build_feature_table()` produces feature rows and validity diagnostics
5. `make_splits()` creates chronological train, validation, and test sets
6. The model is trained and evaluated offline
7. A run bundle is written under `artifacts/runs/consumption/<run_id>/`
8. Exports and summaries are written under `artifacts/exports/consumption/<run_id>/`
9. If promotion is requested, the run bundle is copied to `artifacts/models/consumption/current/`

### Forecast Flow

1. `build_forecast_runtime()` loads the promoted bundle and resolves runtime data inputs
2. History is sliced strictly before the target day
3. A 144-row target-day frame is built at `10min` frequency
4. Forecast features are rebuilt with the same columns stored in the bundle summary
5. The model predicts one value per timestamp
6. Forecast CSV outputs are written under `artifacts/forecasts/consumption/`

### Replay Flow

1. The runtime iterates over each day in the requested range
2. It skips days with incomplete truth coverage
3. Each valid day is forecast through the same strict day-ahead runtime
4. Replay outputs and metrics are written under `artifacts/replays/consumption/`

### API Flow

1. FastAPI routes validate request payloads through Pydantic schemas
2. Service functions call the same training and runtime utilities used by the CLI
3. Async routes submit work to the in-memory thread-pool job manager
4. Results are returned directly or exposed through `/jobs/{job_id}/result`

## Where Do I Change X?

- Change dataset keys or file paths -> `configs/common/data_sources.yaml`, then `src/smartgrid/data/catalog.py` if the contract changes
- Change total target aggregation -> `src/smartgrid/common/constants.py` and `src/smartgrid/data/loaders.py`
- Change split dates -> `configs/consumption/*.yaml`
- Change split semantics -> `src/smartgrid/data/splits.py`
- Add or modify a feature -> `src/smartgrid/features/engineering.py`
- Add a weather feature group -> `src/smartgrid/common/constants.py` and `src/smartgrid/features/engineering.py`
- Change model architecture -> `src/smartgrid/models/mlp.py` and `src/smartgrid/training/trainer.py`
- Add a new model type -> `src/smartgrid/models/`, `src/smartgrid/training/trainer.py`, and `src/smartgrid/registry/model_registry.py`
- Change forecast or replay behavior -> `src/smartgrid/inference/day_ahead.py`
- Change API contracts -> `src/smartgrid/api/schemas.py` and `src/smartgrid/api/app.py`
- Change API business logic -> `src/smartgrid/api/services.py`
- Change notebook orchestration -> `scripts/generate_cli_demo_notebook_v4.py` and `src/smartgrid/notebooks/cli_demo_utils.py`

## Read Next

- [Maintainer Guide](../MAINTAINER_GUIDE.md)
- [Customization Guide](CUSTOMIZATION_GUIDE.md)
- [Data Backend Migration](DATA_BACKEND_MIGRATION.md)
