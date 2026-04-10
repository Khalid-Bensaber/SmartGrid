# Smart Grid

Smart Grid energy forecasting project.

## Current focus
The repo is now centered on a **clean consumption forecasting pipeline**:
- chronological train / validation / test split
- PyTorch MLP baseline with manual lag features
- notebook-aligned metrics and exports
- model registry with explicit promotion to `current`
- API isolated from training artifacts in progress

## Canonical Python package
Use `src/smartgrid/` as the single canonical package.

## Quick start
```bash
uv sync --all-groups
python scripts/train_consumption.py --promote
python scripts/promote_consumption_run.py --run-id <RUN_ID>
uvicorn smartgrid.api:app --reload
```

## Dependency groups
- Core runtime: `uv sync`
- Core + dev tooling: `uv sync --group dev`
- Core + dev + legacy TensorFlow tooling: `uv sync --all-groups`

The `legacy` dependency group is intentionally kept for local development around `src/Legacy/`.
It is not part of the production inference path.

## Main commands
```bash
make install
make install-core
make install-dev
make install-dev-legacy
make test
make train-consumption
make serve-api
```

## Repo layout
- `data/processed`: input datasets only
- `artifacts/`: model files, exports, run summaries, backtests and other generated outputs
- `src/smartgrid/data`: loading and temporal splits
- `src/smartgrid/features`: feature engineering
- `src/smartgrid/models`: PyTorch model definitions
- `src/smartgrid/training`: trainer + artifact management
- `src/smartgrid/evaluation`: metrics + reporting/export
- `src/smartgrid/registry`: publication of validated models
- `src/smartgrid/api`: inference API
- `src/Legacy`: legacy local experimentation kept for dev-only workflows
- `configs/consumption`: experiment configs
- `scripts/`: thin wrappers for local execution
