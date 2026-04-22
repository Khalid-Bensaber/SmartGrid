# Maintainer Guide

Read this file when you need to change SmartGrid rather than only run it. It is the fastest handoff map for the active consumption pipeline.

For deeper explanations, use the companion guides in `docs/`.

## Start From These Entry Points

- Main training orchestration path: `scripts/train_consumption.py` -> `src/smartgrid/cli/train_consumption.py`
- Promoted runtime bundle: `artifacts/models/consumption/current`
- Official operational evaluation path: `scripts/replay_period.py` -> `src/smartgrid/cli/replay_period.py`
- HTTP orchestration layer: `src/smartgrid/api/`

## Active Default Config

The current default training config wired into the Makefile, CLI, and API is:

```bash
configs/consumption/mlp_strict_day_ahead_cyclical_weather_shifted_dynamics_h512_256_128_do010_wd1em5.yaml
```

## Where To Change Data Sources

- Dataset keys and file paths: `configs/common/data_sources.yaml`
- Dataset resolution logic: `src/smartgrid/data/catalog.py`
- CSV loading and target reconstruction: `src/smartgrid/data/loaders.py`
- Setup validation used by `make doctor`: `scripts/check_setup.py`

Use the catalog as the first place to add or swap tracked datasets. Only change loader code when the tabular contract itself changes.

## Where To Change Split Logic

- Split boundaries in the shipped experiments: `configs/consumption/*.yaml`
- Split implementation: `src/smartgrid/data/splits.py`
- Regression tests for boundary semantics: `tests/test_temporal_semantics.py`

The current implementation supports ratio-based splits or date-based `train_end_date` / `val_end_date`. Date-only boundaries include the full day.

## Where To Change Features

- Feature flags and feature construction: `src/smartgrid/features/engineering.py`
- Weather and validity column constants: `src/smartgrid/common/constants.py`
- Training/inference parity tests: `tests/test_day_ahead.py`, `tests/test_features.py`, `tests/test_temporal_semantics.py`

If you add a feature, update both training-time feature generation and forecast-time feature reconstruction. Strict day-ahead mode must keep rejecting features that require same-day target truth.

## Where To Change Model Architecture

- Current model definition: `src/smartgrid/models/mlp.py`
- Training loop and batching behavior: `src/smartgrid/training/trainer.py`
- Bundle loading for prediction and replay: `src/smartgrid/registry/model_registry.py`
- Experiment hyperparameters: `configs/consumption/*.yaml`

The codebase currently assumes a single `TorchMLP` bundle format. If you add another model family, update both training and bundle loading together.

## Where To Change Trainer Behavior

- Optimizer, patience, batching, and device handling: `src/smartgrid/training/trainer.py`
- Scaling and evaluation orchestration: `src/smartgrid/cli/train_consumption.py` and `src/smartgrid/api/services.py`
- Artifact writing and promotion: `src/smartgrid/training/artifacts.py`

Training summaries are persisted in each run bundle and are later reused by inference, replay, notebooks, and the API.

## Where To Change Inference And Replay

- Strict day-ahead runtime, fallback model search, replay loop: `src/smartgrid/inference/day_ahead.py`
- Matrix and row prediction helpers: `src/smartgrid/inference/consumption.py`
- Forecast and replay CLI entry points: `src/smartgrid/cli/predict_next_day.py`, `src/smartgrid/cli/replay_period.py`
- Replay metrics and export helpers: `src/smartgrid/evaluation/reporting.py`, `src/smartgrid/evaluation/metrics.py`

Keep these semantics explicit:

- Forecasting uses the promoted bundle by default.
- Target-day features are rebuilt from history and exogenous inputs.
- Replay is the official operational benchmark.

## Where To Change API Behavior

- Route declarations: `src/smartgrid/api/app.py`
- Request/response schemas: `src/smartgrid/api/schemas.py`
- Business logic: `src/smartgrid/api/services.py`
- Async job manager: `src/smartgrid/api/jobs.py`

The async job manager is in-memory and process-local. It is fine for demos, but it is not a durable scheduler backend.

## Where To Change Notebook And Demo Behavior

- Authoritative demo notebook: `notebooks/experiments/SmartGrid_CLI_Demo_Notebook_v4.ipynb`
- Notebook generator: `scripts/generate_cli_demo_notebook_v4.py`
- Notebook helpers: `src/smartgrid/notebooks/cli_demo_utils.py`

The notebook should stay CLI-first: it should orchestrate repo code, not reimplement the forecasting pipeline in notebook-only logic.

## Recommended Safety Checks After Changes

- `make doctor`
- `make test`
- `make train-consumption CONFIG=configs/consumption/mlp_strict_day_ahead_cyclical_weather_shifted_dynamics_h512_256_128_do010_wd1em5.yaml`
- `make predict-next-day`
- `make replay-period START_DATE=2026-01-01 END_DATE=2026-01-31`

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
