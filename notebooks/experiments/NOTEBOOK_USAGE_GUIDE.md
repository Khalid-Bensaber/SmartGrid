# Notebook Usage Guide

Ce guide accompagne `SmartGrid_CLI_Demo_Notebook_v3.ipynb`.

## Ce qui est officiel vs diagnostic

- `replay` = benchmark métier officiel et classement principal.
- `offline` = diagnostic secondaire issu du split de training.
- `long sample runtime` = validation de la chaîne réelle sur une longue période, utile pour la démo et l'analyse.
- `legacy` = comparaison historique valable seulement jusqu'au `2025-12-10`.

## Sections du notebook

- `A-B` posent le scope, détectent le repo et résolvent les données.
- `D-E` construisent l’inventaire des configs et lancent les entraînements via `make train-consumption`.
- `G-H` lancent ou rechargent le replay officiel puis auditent les `skipped_days`.
- `J-M` produisent et analysent le long sample runtime de `2025-11-20` à `2026-03-19`.
- `O` garde un audit technique `offline vs predict vs replay` sur une date donnée.
- `P-Q` construisent le classement final et peuvent promouvoir le gagnant via la CLI officielle.

## Comment lancer un training

Dans la cellule de configuration :

- mets `RUN_TRAINING = True`
- laisse `CONFIG_PATHS` sur les YAML strict day-ahead voulus
- exécute ensuite la section training

Le notebook appelle `make train-consumption` avec `cwd=ROOT`.

## Comment lancer le replay officiel

Dans la cellule de configuration :

- mets `RUN_REPLAY = True`
- règle `REPLAY_START_DATE` et `REPLAY_END_DATE`
- laisse `REPLAY_MODEL_RUN_IDS = []` pour prendre les derniers runs des configs listées, ou renseigne des `run_id` précis

Le notebook appelle `scripts/benchmark_replay_models.py`.

## Comment lancer le long sample runtime

Dans la cellule de configuration :

- mets `RUN_LONG_SAMPLE_PREDICT = True`
- règle `LONG_SAMPLE_START_DATE = \"2025-11-20\"`
- règle `LONG_SAMPLE_END_DATE = \"2026-03-19\"`
- laisse `LONG_SAMPLE_RUN_IDS = []` pour utiliser le run actif, ou mets une liste explicite de `run_id`

Le notebook appelle `scripts/predict_next_day.py` une fois par jour et met en cache les sorties.

## Où vont les exports

Tous les exports notebook vont sous :

- `artifacts/notebook_exports/cli_demo_v3/`

Fichiers importants :

- `replay_leaderboard_current.csv`
- `skipped_days_audit.csv`
- `long_sample_predict/<run_id>/<start>__<end>/predict_long_sample.csv`
- `long_sample_predict/<run_id>/<start>__<end>/joined_comparison.csv`
- `comparisons/days/model_day_comparison_<date>.csv`
- `comparisons/replay/model_range_comparison_<start>_<end>.csv`
- `comparisons/long_sample/model_range_comparison_<start>_<end>.csv`
- `metadata.json`

## Pourquoi une V3

La V3 garde la V2 comme point de comparaison et de secours.
La refonte est suffisamment profonde pour mériter un nouveau notebook plutôt qu’un remplacement silencieux.
