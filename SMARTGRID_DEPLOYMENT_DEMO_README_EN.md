# SmartGrid  Deployment / Demo README

This guide is designed for a **quick SmartGrid project demonstration** using Docker.

> Important assumption  
> This guide assumes that the **corrected Docker files** are already present at the root of the repository:
> - `Dockerfile`
> - `docker-compose.yml`
> - `docker-compose.gpu.yml` (optional, only for GPU machines)
> - `docker/entrypoint.sh`
> - `.dockerignore`

---

## 1. Goal

The goal is to be able to:
- clone the repository,
- start a working container,
- verify the environment,
- train a model,
- promote a model,
- run a simple prediction,
- optionally start the API.

---

## 2. Requirements

Host machine:
- Git
- Docker + Docker Compose

Optional:
- NVIDIA GPU + NVIDIA runtime if you want to run PyTorch on CUDA inside the container

---

## 3. Get the project

```bash
git clone -b dev https://github.com/Khalid-Bensaber/SmartGrid.git
cd SmartGrid
```

If needed, then place the corrected Docker files at the root of the repository.

---

## 4. Choose the mode: CPU or GPU

### CPU mode / machine without GPU

Use only the main Compose file:

```bash
docker compose build cli --progress=plain
docker compose run --rm cli
```

### GPU mode / machine with NVIDIA GPU

Use the GPU override:

```bash
docker compose build cli --progress=plain
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm cli
```

### Start the API

CPU:

```bash
docker compose up api
```

GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up api
```

---

## 5. Basic checks inside the container

Once you are inside the container shell:

```bash
pwd
python --version
uv --version
make --version
ls
ls data/processed
ls artifacts
```

Project health check:

```bash
make doctor
make verify
```

GPU check (GPU machine only):

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

---

## 6. Training

### Simple baseline training

```bash
make train-consumption
```

### Training with a specific config

```bash
make train-consumption CONFIG=configs/consumption/mlp_strict_day_ahead_baseline.yaml ANALYSIS_DAYS=1 DATASET_KEY=full_2020_2026
```

### Training + immediate promotion

This is the most practical command for the demo because it avoids forgetting the promotion step:

```bash
make train-promote CONFIG=configs/consumption/mlp_strict_day_ahead_baseline.yaml ANALYSIS_DAYS=1 DATASET_KEY=full_2020_2026
```

---

## 7. Promote an existing run

If a training run was already completed without promotion, promote it explicitly:

```bash
make promote-consumption RUN_ID=consumption_mlp_YYYYMMDDTHHMMSSZ
```

Tip:
- the `RUN_ID` is displayed in the final JSON output of the training script;
- after promotion, the current bundle should exist in:

```bash
ls artifacts/models/consumption/current
```

---

## 8. Prediction

### Predict a specific day

```bash
make predict-next-day TARGET_DATE=2026-01-15
```

### Direct script call

```bash
uv run python scripts/predict_next_day.py --target-date 2026-01-15
```

### Important

Prediction **loads the promoted current model**.  
So if `artifacts/models/consumption/current/model.pt` does not exist, prediction will fail.

In practice:
- either use `make train-promote`,
- or run `make train-consumption` and then `make promote-consumption RUN_ID=...`.

---

## 9. Historical replay

To replay a historical period:

```bash
make replay-period START_DATE=2026-01-01 END_DATE=2026-01-31
```

Minimal example on a single day:

```bash
make replay-period START_DATE=2026-01-15 END_DATE=2026-01-15
```

---

## 10. API

Start the API:

CPU:

```bash
docker compose up api
```

GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up api
```

Then open:

- `http://localhost:8000/docs`
- `http://localhost:8000/health`

---

## 11. Notebook

Optional.

CPU:

```bash
docker compose --profile notebook up notebook
```

GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile notebook up notebook
```

---

## 12. File persistence

The repository is mounted into the container with `.:/workspace`.

So the following remain persistent between runs:
- `artifacts/`
- `data/raw/`
- `data/interim/`
- `data/external/`
- `data/processed/`
- the code, notebooks, configs, and scripts

---

## 13. Recommended demo sequence

### CPU demo

```bash
git clone -b dev https://github.com/Khalid-Bensaber/SmartGrid.git
cd SmartGrid
docker compose build cli --progress=plain
docker compose run --rm cli
```

Then inside the container:

```bash
make doctor
make verify
make train-promote CONFIG=configs/consumption/mlp_strict_day_ahead_baseline.yaml ANALYSIS_DAYS=1 DATASET_KEY=full_2020_2026
make predict-next-day TARGET_DATE=2026-01-15
make replay-period START_DATE=2026-01-15 END_DATE=2026-01-15
```

### GPU demo

```bash
git clone -b dev https://github.com/Khalid-Bensaber/SmartGrid.git
cd SmartGrid
docker compose build cli --progress=plain
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm cli
```

Then inside the container:

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
make doctor
make train-promote CONFIG=configs/consumption/mlp_strict_day_ahead_baseline.yaml ANALYSIS_DAYS=1 DATASET_KEY=full_2020_2026
make predict-next-day TARGET_DATE=2026-01-15
```

---

## 14. Quick troubleshooting

### Error: `artifacts/models/consumption/current/model.pt` not found

Cause:
- the model was trained,
- but not promoted.

Solution:

```bash
make train-promote CONFIG=configs/consumption/mlp_strict_day_ahead_baseline.yaml ANALYSIS_DAYS=1 DATASET_KEY=full_2020_2026
```

or:

```bash
make promote-consumption RUN_ID=consumption_mlp_YYYYMMDDTHHMMSSZ
```

### The GPU is not visible

Check that:
- the host machine can already see the GPU,
- the `docker-compose.gpu.yml` override is actually being used,
- PyTorch returns `torch.cuda.is_available() == True`.

### The Docker build is very slow

Use:

```bash
docker compose build cli --progress=plain
```

and build only `cli` instead of all services.

---

## 15. Reference commands

```bash
make doctor
make verify
make train-consumption
make train-promote CONFIG=configs/consumption/mlp_strict_day_ahead_baseline.yaml ANALYSIS_DAYS=1 DATASET_KEY=full_2020_2026
make promote-consumption RUN_ID=consumption_mlp_YYYYMMDDTHHMMSSZ
make predict-next-day TARGET_DATE=2026-01-15
make replay-period START_DATE=2026-01-01 END_DATE=2026-01-31
docker compose up api
docker compose --profile notebook up notebook
```
