# Customization Guide

Audience: maintainers and future project owners who need to extend SmartGrid safely.

The current production-style implementation is consumption-first and strict day-ahead-first. This guide explains where to change that behavior without breaking training and runtime parity.

## Before You Change Anything

- Start from the active path in `scripts/train_consumption.py`
- Keep replay as the official evaluation path
- Preserve the promoted bundle convention under `artifacts/models/consumption/current`
- Update tests whenever you change time semantics, features, or API behavior

## Change The Training Date Boundaries

Current behavior:

- the shipped configs define `train_end_date` and `val_end_date`
- `src/smartgrid/data/splits.py` applies date-only boundaries as full-day inclusive boundaries
- if both end dates are absent, the code falls back to ratio-based splitting

What to edit:

- the `split:` section in `configs/consumption/*.yaml`
- `src/smartgrid/data/splits.py` if you need new boundary semantics
- `tests/test_temporal_semantics.py` for regression coverage

Recommended validation:

- run `make test`
- run one `make train-consumption ...` command and confirm train/val/test ranges in the run summary

## Add `train_start_date`

Current limitation:

- the split layer supports `train_end_date` and `val_end_date`, but not an explicit training start date

Recommended implementation path:

1. Add `train_start_date` to the `split:` section of the YAML configs that need it.
2. Extend `make_splits()` and `chronological_split_by_dates()` in `src/smartgrid/data/splits.py`.
3. Apply an additional lower-bound mask before constructing the train split.
4. Keep the existing date-only versus timestamp-exact behavior consistent with the current tests.
5. Add tests for both date-only and timestamp-level `train_start_date` semantics in `tests/test_temporal_semantics.py`.

Important: avoid silently changing the meaning of existing configs that do not provide a start date.

## Forecast One Building Instead Of The Total Aggregation

Current behavior:

- `load_history()` in `src/smartgrid/data/loaders.py` reconstructs `tot` by summing the four columns listed in `TOTAL_COLUMNS`
- if `target_col != "tot"`, the current implementation still writes the total into that requested target column

That means the current codebase is hard-wired to total-consumption training.

Recommended change path:

1. Decide whether you want a single building target such as `Ptot_HA` or a configurable target strategy.
2. Refactor `load_history()` so it preserves raw building columns and only computes `tot` when the config actually asks for it.
3. Keep `tot` available if you still need total-based comparison or notebooks.
4. Update the `data.target_name` field in the relevant YAML configs.
5. Re-run training and replay on the new target.

Impact to check:

- `load_old_benchmark()` currently produces a total-only legacy forecast comparison
- notebook exports and reporting helpers assume total forecast naming
- any API responses still labeled `Ptot_TOTAL_Forecast` will need renaming or a target-aware schema

## Change The Aggregation Logic

Current behavior:

- the total target is hard-coded as the sum of `TOTAL_COLUMNS` in `src/smartgrid/common/constants.py`
- partial totals are rejected by requiring all source building columns to be present

If you need a different aggregation rule:

1. Refactor the aggregation code in `src/smartgrid/data/loaders.py` into a named helper instead of editing the sum inline.
2. Decide whether missing source columns should fail hard, produce `NA`, or use a different policy.
3. Update any documentation or benchmark assumptions that still talk about “total” semantics.

Do not silently change the current `tot` meaning without also retraining every affected bundle.

## Change The Set Of Lag Days

Current behavior:

- lag days live in `features.lag_days` inside the YAML configs
- both training and forecast reconstruction read the same lag list from bundle metadata

What to edit:

- `configs/consumption/*.yaml`
- tests in `tests/test_features.py` or `tests/test_temporal_semantics.py` when the semantics change materially

Validation:

- retrain
- run `make predict-next-day`
- run `make replay-period`

## Add A New Feature

Rules for a safe feature addition:

1. Add the config flag or config parameters in `normalize_feature_config()` inside `src/smartgrid/features/engineering.py`.
2. Compute the feature in `build_feature_table()` for training.
3. Recompute the same feature in `build_forecast_feature_row()` or `build_temporal_feature_values()` for inference.
4. Decide whether the feature should participate in validity filtering.
5. Add or update tests proving training/inference parity.

Strict day-ahead warning:

- if the feature depends on same-day target observations, it must not be allowed in `strict_day_ahead`

Helpful test locations:

- `tests/test_day_ahead.py`
- `tests/test_features.py`
- `tests/test_temporal_semantics.py`

## Add Weather Features

Current behavior:

- weather column groups are defined in `src/smartgrid/common/constants.py`
- feature selection happens through `resolve_weather_columns()` in `src/smartgrid/features/engineering.py`

To add another weather mode:

1. add the renamed columns to the constants file
2. extend `resolve_weather_columns()`
3. ensure `load_weather_history()` and `attach_exogenous_columns()` provide the needed values
4. add a config that enables the new weather mode

## Change Trainer Behavior

Configuration-level changes:

- `training.hidden_layers`
- `training.learning_rate`
- `training.weight_decay`
- `training.batch_size`
- `training.epochs`
- `training.patience`
- `training.dropout`
- `training.device`

Code-level changes:

- `src/smartgrid/training/trainer.py` for optimizer choice, batching behavior, and early stopping
- `src/smartgrid/cli/train_consumption.py` and `src/smartgrid/api/services.py` for the orchestration around scaling and evaluation

If you change trainer behavior, keep the run summary complete enough for later replay and API inspection.

## Add A New Model

Current behavior:

- the codebase assumes `TorchMLP` for both save and load paths
- bundle loading reconstructs that model directly from `model_config`

Recommended extension path:

1. Add the new model class under `src/smartgrid/models/`.
2. Introduce a `model_type` selection path in training, ideally from config.
3. Save enough metadata in `model_config` for bundle reconstruction.
4. Update `src/smartgrid/registry/model_registry.py` so bundle loading is model-type aware.
5. Add at least one config file that selects the new model.
6. Re-check API model-info and notebook assumptions if they currently assume only MLP metadata.

## Add Production Forecasting Alongside Consumption

Current repository state:

- consumption is the only production-style CLI/API pipeline
- PV data assets exist under `data/processed/prod/`, but there is no parallel operational runtime

Recommended structure:

- `configs/production/` for production experiments
- `src/smartgrid/inference/production.py`
- `src/smartgrid/cli/train_production.py`, `predict_production.py`, `replay_production.py`
- `artifacts/runs/production/`, `artifacts/models/production/`, `artifacts/replays/production/`
- `/production/...` API routes alongside `/consumption/...`

Prefer target-specific modules over filling the current consumption code with special cases everywhere.

## Extend The API For New Behavior

What to edit:

- request or response shapes: `src/smartgrid/api/schemas.py`
- routes: `src/smartgrid/api/app.py`
- service implementation: `src/smartgrid/api/services.py`
- async job handling only if the new behavior needs background execution: `src/smartgrid/api/jobs.py`

Keep the API aligned with the CLI whenever possible so notebook and scheduler usage stay consistent.

## Add Scheduler-Driven Automation

Recommended pattern:

1. use the API or CLI to run training
2. evaluate replay outputs
3. promote only after your acceptance criteria pass
4. call forecast on the promoted bundle

Current limitation:

- async API jobs are process-local and in-memory, so they do not survive restarts or scale across multiple API workers

## Test Checklist After Customization

- `make test`
- `make train-consumption CONFIG=<your-config>`
- `make predict-next-day TARGET_DATE=2026-01-15`
- `make replay-period START_DATE=2026-01-01 END_DATE=2026-01-31`

## Read Next

- [Architecture and Code Map](ARCHITECTURE_AND_CODE_MAP.md)
- [Data Backend Migration](DATA_BACKEND_MIGRATION.md)
- [API and Scheduler Integration](API_AND_SCHEDULER_INTEGRATION.md)
