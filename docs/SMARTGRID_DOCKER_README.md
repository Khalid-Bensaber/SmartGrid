# SmartGrid Docker Pack

This pack is designed for the current `dev` branch workflow:
- Python 3.12
- `uv` as the package manager / runner
- `make` as the main command entry point
- processed datasets versioned in `data/processed/`
- generated outputs written under `artifacts/`

## What this pack does

### Dockerfile
Builds a Linux image that already contains:
- Python 3.12
- `bash`
- `build-essential`
- `curl`
- `git`
- `make`
- `tini`
- `uv`

It copies the repo into `/workspace`, creates the local runtime directories that should always exist, and runs `uv sync --all-groups` so the project is ready to use.

### docker/entrypoint.sh
Runs every time a container starts.
It:
1. moves into `/workspace`
2. ensures `artifacts/`, `data/raw/`, `data/interim/`, and `data/external/` exist
3. refreshes the environment with `uv sync`
4. runs `make doctor` by default
5. optionally runs `make verify`
6. launches the final command (`bash`, `make serve-api`, etc.)

### docker-compose.yml
Provides three services:
- `cli`: interactive shell for `make` commands and scripts
- `api`: starts the FastAPI service on port `8000`
- `notebook`: optional JupyterLab service on port `8888`

The compose file bind-mounts the whole repo into `/workspace`.
That is the recommended default because it preserves **all local runtime outputs** in the repository folder on the host machine, including:
- `artifacts/`
- `data/raw/`
- `data/interim/`
- `data/external/`

### .dockerignore
Prevents Docker builds from sending useless or sensitive local files into the build context:
- virtual environments
- caches
- ignored local data
- generated artifacts
- editor files
- `.env*`

It intentionally does **not** ignore `data/processed/`, because the current repo design expects those processed datasets to be available on a fresh clone.

## Why the whole repo is mounted

The current repo tracks the code and `data/processed/`, while generated outputs stay local.
By mounting `.:/workspace`, you keep everything that matters on the host filesystem:
- code changes
- notebooks
- processed data
- artifacts
- extra local data folders

This is the safest and simplest setup for development and day-to-day use.

## Commands

### Build the image
```bash
docker compose build
```

### Open a shell in the prepared environment
```bash
docker compose run --rm cli
```

Then inside the container:
```bash
make doctor
make verify
make train-consumption
make train-promote CONFIG=configs/consumption/mlp_strict_day_ahead_cyclical_weather_shifted_dynamics.yaml ANALYSIS_DAYS=3
make predict-next-day TARGET_DATE=2026-01-15
make replay-period START_DATE=2026-01-01 END_DATE=2026-01-31
```

### Start the API
```bash
docker compose up api
```

### Start JupyterLab
```bash
docker compose --profile notebook up notebook
```

## Host requirements

The host machine only needs a working Docker runtime able to run Linux containers.
The host does **not** need:
- Python
- `uv`
- `make`
- `git`
- `apt`

Those tools are installed **inside the image**.
