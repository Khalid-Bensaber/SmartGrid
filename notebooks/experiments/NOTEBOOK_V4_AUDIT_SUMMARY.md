# NOTEBOOK V4 Audit Summary

## What Was Restructured

`SmartGrid_CLI_Demo_Notebook_v4.ipynb` recenters the notebook around the **official long-sample replay benchmark** instead of a short replay window or a winner-only demo flow.

The notebook is now organized around:

1. scope and semantics
2. inventory of configs and runs
3. official long-sample replay benchmark
4. skipped-days audit
5. long-sample multi-model study
6. legacy overlap study
7. errors and bias study
8. protocol audit
9. final ranking across periods
10. predict/demo export
11. promotion

## What Was Removed Or De-emphasized

- The notebook is no longer centered on a single-model long-sample runtime demo.
- The protocol audit remains present, but it is no longer treated like a benchmark section.
- The offline diagnostic is still visible, but it is explicitly secondary to replay.

## What Was Added

- Official replay leaderboard on `2025-11-20 -> 2026-03-19`
- Multi-model long-sample comparison frame
- Per-model and per-day long-sample metrics exports
- Legacy overlap metrics export on `2025-11-20 -> 2025-12-10`
- Ranking comparison across:
  - long sample official
  - short-term reference
  - legacy overlap
  - 2026 subperiod
- Skipped-days alignment audit
- Worst-days analysis
- Transition-window analysis for late 2025 to early 2026
- Predict/demo export separated from the official benchmark

## How The Winner Is Now Chosen

The official winner is now chosen from the **replay benchmark on the long official window**:

- start: `2025-11-20`
- end: `2026-03-19`

This replaces the earlier logic where a winner could be inferred from a shorter replay window.

The notebook still computes secondary rankings on shorter or more specific periods, but those do **not** override the long-sample official winner.

## Why This Is Better

- It evaluates model stability over a much longer horizon.
- It includes both easier late-2025 behavior and harder 2026 behavior.
- It makes skipped days visible instead of letting them hide inside summary files.
- It gives actual study graphs for multi-model comparison rather than only showcasing one winner.

## Main Exports Produced

Under `artifacts/notebook_exports/cli_demo_v4/` the notebook now writes:

- `long_sample_replay_leaderboard.csv`
- `model_range_comparison_long_sample.csv`
- `long_sample_metrics_by_model.csv`
- `per_day_metrics_by_model_long_sample.csv`
- `legacy_overlap_metrics_by_model.csv`
- `period_ranking_comparison.csv`
- `worst_days_by_model_long_sample.csv`
- `skipped_days_audit.csv`
- `metadata.json`

## Helper Changes

The notebook helper module `src/smartgrid/notebooks/cli_demo_utils.py` was extended with:

- `compute_series_metrics(...)`
- `compute_series_daily_metrics(...)`
- `compute_series_error_frame(...)`
- generic series-column helpers for multi-model analysis

These additions keep the notebook readable while avoiding hidden training or replay logic inside notebook-only code.

## Remaining Limits

- The notebook still depends on existing replay artifacts unless `RUN_REPLAY_LONG_SAMPLE=True`.
- Plot density can become high if too many models are selected at once; the notebook therefore defaults to a `top K` overlay approach for visual sections.
- The protocol audit is still a local guardrail on one run and one date; it is not a global proof over all runs and all dates.
- If artifacts are stale relative to code changes, users should regenerate replay outputs before making decisions.

## Practical Recommendation

Use V4 as the main benchmark notebook.

- Use the **official long-sample replay** for winner selection.
- Use the **legacy overlap** section to judge historical comparability.
- Use the **errors and bias** section to understand robustness, underprediction, and worst days.
- Use the **predict/demo** section only for runtime demonstration of a chosen run.
