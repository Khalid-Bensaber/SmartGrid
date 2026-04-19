# SmartGrid Docker + .gitignore audit

## What the repository already expects

The current `dev` branch is already organized around:

- Python `3.12`
- `uv` as the package/dependency manager
- `make` as the operational entry point
- a FastAPI app started with `make serve-api`
- training, promotion, prediction, replay and benchmark commands exposed through the `Makefile`

So the container should follow that contract instead of bypassing it.

## Why the current docker-compose is too weak

The repository currently ships a very small compose file that:

- uses `python:3.12-slim`
- bind-mounts the whole repo
- runs `pip install -e .`
- starts `uvicorn smartgrid.api:app`

That is not enough for this repo because:

1. the project workflow is based on `uv sync`, not plain `pip install -e .`
2. the repo README expects `make verify` / `make doctor`
3. the repo uses a custom `uv` torch index for CUDA wheels in `pyproject.toml`
4. there is no dedicated CLI shell service
5. persistence is not thought through explicitly

## Persistence strategy

For **this repository**, the simplest and safest persistence strategy is a **bind mount of the whole repo**:

- `.:/workspace`

This preserves:

- tracked code and configs
- tracked `data/processed/`
- ignored local outputs like `artifacts/`
- anything you edit from inside the container

### Important caveat

Do **not** mount a separate named volume over `/workspace/data/processed`, because that would hide the processed datasets that are intentionally tracked in Git.

If later you want image-only runtime without a full bind mount, then the directories that deserve dedicated persistence are mostly:

- `/workspace/artifacts`
- `/workspace/data/raw`
- `/workspace/data/interim`
- `/workspace/data/external`

## .gitignore audit: what is shared vs what must be recreated/shared manually

### Already shared through Git

These are explicitly part of the repo according to the README and current layout:

- `src/`
- `scripts/`
- `configs/`
- `tests/`
- `notebooks/`
- `data/processed/`
- `pyproject.toml`
- `uv.lock`
- `README.md`
- `Makefile`

### Intentionally local / ignored

These are ignored and therefore must be **recreated, copied, or shared separately if needed**:

- `.venv/`, `venv/`, `env/`
- `.env`, `.env.*` except `.env.example`
- `data/raw/*`
- `data/interim/*`
- `data/external/*`
- `artifacts/`
- `logs/`
- `runs/`
- `mlruns/`
- `tensorboard/`
- editor/cache/temp files

### Practical consequence for a new machine or server

If the clone contains the tracked repo files, the missing pieces are usually only:

- local environment variables
- raw/intermediate/external data you intentionally keep outside Git
- generated model outputs under `artifacts/`
- any local experiment logs

## Recommended usage

### Build

```bash
docker compose build
```

### API

```bash
docker compose up api
```

### Open a shell for CLI work

```bash
docker compose run --rm cli
```

### Examples inside the CLI container

```bash
make doctor
make verify
make train-consumption
make train-promote CONFIG=configs/consumption/mlp_strict_day_ahead_cyclical_weather_shifted_dynamics.yaml ANALYSIS_DAYS=3
make predict-next-day TARGET_DATE=2026-01-15
make replay-period START_DATE=2026-01-01 END_DATE=2026-01-31
```
