# Notebook and Demo Guide

Audience: presenters, evaluators, and teammates preparing an oral demo or benchmark walkthrough.

This guide explains which notebook to trust, which files are generated versus historical, and what is safe to show live.

## Authoritative Notebook

The authoritative demo notebook is:

- `notebooks/experiments/SmartGrid_CLI_Demo_Notebook_v4.ipynb`

Why this one:

- it is aligned with the current `dev` branch
- it treats replay as the official benchmark
- it separates offline diagnostics from operational replay semantics
- it exports reusable outputs under `artifacts/notebook_exports/cli_demo_v4/`

If you need to rebuild it, use:

```bash
uv run python scripts/generate_cli_demo_notebook_v4.py
```

## Generated Vs Manual Notebook Files

Generated and currently authoritative:

- `SmartGrid_CLI_Demo_Notebook_v4.ipynb`
- generated from `scripts/generate_cli_demo_notebook_v4.py`

Manual or historical notebooks:

- older `SmartGrid_CLI_Demo_Notebook*.ipynb` files
- `SmartGrid_Demo_Globale*.ipynb`
- exploratory analysis notebooks such as `SmartGrid_Replay_Benchmark_Analyse.ipynb` and `compare_consumption_feature_variants.ipynb`

Rule of thumb:

- use V4 for the official demo story
- treat older notebooks as historical analysis material, not operational truth

## What To Precompute Before The Oral Demo

Minimum recommended preparation:

1. `make doctor`
2. `make verify`
3. ensure a promoted model exists with `make train-promote` or `make promote-consumption RUN_ID=...`
4. run at least one replay window and keep the outputs
5. open the V4 notebook and confirm it can load the existing artifacts

If you want the notebook’s long official benchmark to be ready without recalculation, precompute:

- a replay benchmark covering `2025-11-20` to `2026-03-19`
- the exports under `artifacts/notebook_exports/cli_demo_v4/`

Useful commands:

```bash
make train-promote
make replay-period START_DATE=2025-11-20 END_DATE=2026-03-19
uv run python scripts/generate_cli_demo_notebook_v4.py
```

If you want multi-model comparison in the notebook, also precompute benchmark outputs with:

```bash
make benchmark-replay MODEL_REFS='run_a run_b'
```

## Safe Live Steps To Show

Safe and fast:

- open the README and explain the project scope
- run `make doctor`
- run `make predict-next-day TARGET_DATE=2026-01-15`
- open `http://localhost:8000/docs` after `make serve-api`
- show replay outputs that were computed ahead of time
- walk through the V4 notebook with cached exports already present

Usually safe only on a prepared machine:

- `make replay-period` on a short window
- API forecast requests

Usually not safe live unless you already tested the exact environment:

- full retraining during the presentation
- long replay benchmarks
- ad hoc notebook refactoring
- any workflow that depends on rebuilding large artifacts from scratch

## Demo Narrative Recommendation

1. Show the repository entry point in `README.md`.
2. Explain that the active operational scope is consumption forecasting.
3. Show the promoted-model workflow: train, promote, predict.
4. Emphasize that replay is the official operational benchmark.
5. Use the V4 notebook to show benchmark results, skipped-day handling, and model comparisons.
6. Use the API docs page only as a short proof that the same workflow is available over HTTP.

## How Notebook Outputs Relate To Replay Outputs

The V4 notebook is intentionally CLI-first.

- It should consume or orchestrate repo commands rather than reimplement the model pipeline.
- Replay outputs remain the operational source of truth.
- Notebook exports are presentation and analysis artifacts built on top of those repo outputs.

That separation matters during handoff:

- replay metrics decide operational quality
- notebook visuals explain and communicate those results

## Practical Files To Keep Ready

- the current promoted bundle in `artifacts/models/consumption/current/`
- at least one recent replay directory under `artifacts/replays/consumption/`
- notebook exports under `artifacts/notebook_exports/cli_demo_v4/`
- logs under `artifacts/logs/` in case you need to explain what was run

## What To Avoid Improvising Live

- changing feature configs on the spot
- modifying split logic during the demo
- retraining several models from scratch
- depending on async API jobs if the API process might restart
- using older notebooks as if they described the current pipeline

## Read Next

- [Quick Start](QUICKSTART.md)
- [Operations and Deployment](OPERATIONS_AND_DEPLOYMENT.md)
