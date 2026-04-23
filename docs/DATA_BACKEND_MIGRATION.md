# Data Backend Migration

Audience: future maintainers replacing the current CSV-backed loading layer with Cassandra or another backend.

The goal is to change storage access without changing the downstream forecasting semantics.

## Current Principle

SmartGrid expects normalized tabular inputs with stable semantics. The safest migration is:

1. keep the downstream interface dataframe-based
2. replace the upstream storage access layer
3. preserve timestamps, column names, and target semantics

Do not rewrite feature engineering, training, inference, replay, and API logic around Cassandra directly if you can avoid it.

## Where CSV Enters Today

### Dataset Resolution

- `configs/common/data_sources.yaml`
- `src/smartgrid/data/catalog.py`

This layer resolves dataset keys such as `full_2020_2026` into concrete file paths.

### History Loading

- `src/smartgrid/data/loaders.py::load_history()`

Current responsibilities:

- read the historical CSV
- sort and validate timestamps
- reject duplicate timestamps
- reconstruct `tot` from the four building columns in `TOTAL_COLUMNS`
- attach `timeline_diagnostics`
- ensure `Airtemp` exists even if weather history is separate

### Weather Loading

- `src/smartgrid/data/loaders.py::load_weather_history()`
- `src/smartgrid/data/loaders.py::attach_exogenous_columns()`

Current responsibilities:

- load the weather CSV
- rename weather fields to `Weather_*`
- merge them by timestamp
- interpolate and forward/back fill weather columns after the merge

### Holidays And Benchmark Inputs

- `src/smartgrid/data/loaders.py::load_holiday_sets()`
- `src/smartgrid/data/loaders.py::load_old_benchmark()`

These remain part of the broader data contract even if history eventually comes from Cassandra.

## Downstream Contracts You Must Preserve

- `Date` remains parseable as a pandas datetime column
- rows remain at `10min` cadence for the current consumption pipeline
- duplicate timestamps remain invalid
- `tot` means the total target used for training and replay
- `Airtemp` stays distinct from `Weather_AirTemp`
- feature engineering still receives a sorted dataframe
- inference can still slice history strictly before the target day
- replay can still verify complete target-day truth coverage

Important current subtlety:

- total target reconstruction uses `min_count=len(TOTAL_COLUMNS)`, so partial building coverage yields `NA` rather than a partial total

If you change that behavior silently, replay and evaluation results will change.

## Recommended Migration Shape

Create a backend abstraction at the loader boundary.

Example shape:

```python
class HistoryRepository:
    def load_history(self, ...): ...
    def load_weather(self, ...): ...
    def load_benchmark(self, ...): ...

class CsvHistoryRepository(HistoryRepository):
    ...

class CassandraHistoryRepository(HistoryRepository):
    ...
```

Then make the current loader functions call the selected repository and normalize the returned dataframes into the same contract the rest of the system already expects.

## Where To Insert Cassandra Access

Best insertion points:

- `src/smartgrid/data/loaders.py` for the repository boundary
- optionally `src/smartgrid/data/catalog.py` if dataset keys must also select a backend or keyspace

Avoid pushing backend-specific code into:

- `src/smartgrid/features/engineering.py`
- `src/smartgrid/inference/day_ahead.py`
- `src/smartgrid/training/trainer.py`
- `src/smartgrid/api/services.py`

Those layers should stay backend-agnostic.

## Cassandra Normalization Checklist

- convert query results into pandas dataframes
- parse timestamps exactly once
- sort ascending by `Date`
- remove or reject duplicates consistently
- preserve raw building columns if total reconstruction still happens downstream
- provide weather columns in the same renamed `Weather_*` form
- keep missingness behavior explicit instead of silently filling target values

## What Must Not Change Silently

- timestamp granularity
- timezone or timestamp interpretation
- total-target definition
- weather column naming
- holiday handling
- benchmark alignment rules
- the strict split between historical context and target-day truth

## Parity Tests Before Switching

Before changing the default backend, compare CSV and Cassandra results on the same date windows.

- row counts
- min and max timestamps
- duplicate count
- total target values
- missingness patterns
- weather column presence
- one `make predict-next-day` run
- one `make replay-period` run

Useful existing regression anchors:

- `tests/test_data_catalog.py`
- `tests/test_temporal_semantics.py`
- `tests/test_day_ahead.py`

## Rollout Recommendation

1. Add Cassandra as an optional backend.
2. Keep CSV as the reference implementation during parity testing.
3. Compare replay metrics for the same promoted bundle on both backends.
4. Switch the default only after parity is acceptable and documented.

## Bottom Line

The migration target is not “teach every module about Cassandra.”

The migration target is “replace storage access while preserving SmartGrid’s current dataframe contract and strict day-ahead semantics.”

Most of the data is injected via CSV files into pandas meaning that the only line that must change to add the cassandra connexion is the import of that csv into pands. The only big requirements is that it has to be the same format as the csv.

```python
df = pd.read_csv(csv_path)
 ```

## Documentation Index

- [README.md](../README.md)
- [docs/QUICKSTART.md](QUICKSTART.md)
- [docs/OPERATIONS_AND_DEPLOYMENT.md](OPERATIONS_AND_DEPLOYMENT.md)
- [docs/API_AND_SCHEDULER_INTEGRATION.md](API_AND_SCHEDULER_INTEGRATION.md)
- [docs/ARCHITECTURE_AND_CODE_MAP.md](ARCHITECTURE_AND_CODE_MAP.md)
- [docs/CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md)
- [docs/DATA_BACKEND_MIGRATION.md](DATA_BACKEND_MIGRATION.md)
- [docs/NOTEBOOK_AND_DEMO_GUIDE.md](NOTEBOOK_AND_DEMO_GUIDE.md)
- [MAINTAINER_GUIDE.md](../MAINTAINER_GUIDE.md)
