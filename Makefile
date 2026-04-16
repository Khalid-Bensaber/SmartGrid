.DEFAULT_GOAL := help

.PHONY: help install install-core install-dev install-dev-legacy doctor lint test build verify \
	notebook serve-api train-consumption train-promote promote-consumption \
	predict-next-day replay-period benchmark-features benchmark-replay

UV ?= uv
PYTHON ?= python
RUN = $(UV) run $(PYTHON)

CONFIG ?= configs/consumption/mlp_baseline.yaml
ANALYSIS_DAYS ?= 1
TARGET_DATE ?= 2026-01-15
START_DATE ?= 2026-01-01
END_DATE ?= 2026-01-31
RUN_ID ?=
MODEL_REFS ?=

HISTORICAL_CSV ?= data/processed/conso/Consumption data 2020-2026.csv
BENCHMARK_CSV ?= data/processed/conso/Consumption forecast 2020-2026.csv
WEATHER_CSV ?= data/processed/Weather data 2020-2026.csv
HOLIDAYS_XLSX ?= data/processed/Holidays.xlsx

API_HOST ?= 0.0.0.0
API_PORT ?= 8000

BENCHMARK_CONFIGS ?= \
	configs/consumption/mlp_baseline.yaml \
	configs/consumption/mlp_weather_basic.yaml \
	configs/consumption/mlp_weather_all.yaml

help:
	@echo "Smart Grid make targets"
	@echo "  make install                 Install all dependency groups with uv"
	@echo "  make install-core            Install only runtime dependencies"
	@echo "  make install-dev             Install the dev dependency group"
	@echo "  make install-dev-legacy      Install all groups, including legacy extras"
	@echo "  make doctor                  Check required dataset files from the catalog"
	@echo "  make lint                    Run Ruff on src/, tests/, and scripts/"
	@echo "  make test                    Run the test suite with uv"
	@echo "  make build                   Build sdist and wheel"
	@echo "  make verify                  Run doctor, tests, and build"
	@echo "  make notebook                Start JupyterLab"
	@echo "  make serve-api               Start the FastAPI server"
	@echo "  make train-consumption       Train with CONFIG=<yaml>"
	@echo "  make train-promote           Train and immediately promote the run"
	@echo "  make promote-consumption     Promote RUN_ID=<consumption_mlp_...>"
	@echo "  make predict-next-day        Forecast TARGET_DATE=<YYYY-MM-DD>"
	@echo "  make replay-period           Replay START_DATE=<...> END_DATE=<...>"
	@echo "  make benchmark-features      Compare BENCHMARK_CONFIGS=<...>"
	@echo "  make benchmark-replay        Compare MODEL_REFS='run1 run2 ...'"

install:
	$(UV) sync --all-groups

install-core:
	$(UV) sync

install-dev:
	$(UV) sync --group dev

install-dev-legacy:
	$(UV) sync --all-groups

doctor:
	$(RUN) scripts/check_setup.py

lint:
	$(UV) run ruff check src tests scripts

test:
	$(UV) run pytest

build:
	$(UV) build

verify:
	$(MAKE) doctor
	$(MAKE) test
	$(MAKE) build

notebook:
	$(UV) run jupyter lab

serve-api:
	$(UV) run uvicorn smartgrid.api:app --reload --host $(API_HOST) --port $(API_PORT)

train-consumption:
	$(RUN) scripts/train_consumption.py --config "$(CONFIG)" --analysis-days "$(ANALYSIS_DAYS)"

train-promote:
	$(RUN) scripts/train_consumption.py --config "$(CONFIG)" --analysis-days "$(ANALYSIS_DAYS)" --promote

promote-consumption:
	@test -n "$(RUN_ID)" || (echo "RUN_ID is required. Example: make promote-consumption RUN_ID=consumption_mlp_..."; exit 1)
	$(RUN) scripts/promote_consumption_run.py --run-id "$(RUN_ID)"

predict-next-day:
	$(RUN) scripts/predict_next_day.py \
		--historical-csv "$(HISTORICAL_CSV)" \
		--weather-csv "$(WEATHER_CSV)" \
		--holidays-xlsx "$(HOLIDAYS_XLSX)" \
		--target-date "$(TARGET_DATE)"

replay-period:
	$(RUN) scripts/replay_period.py \
		--historical-csv "$(HISTORICAL_CSV)" \
		--weather-csv "$(WEATHER_CSV)" \
		--holidays-xlsx "$(HOLIDAYS_XLSX)" \
		--start-date "$(START_DATE)" \
		--end-date "$(END_DATE)"

benchmark-features:
	$(RUN) scripts/benchmark_feature_variants.py $(BENCHMARK_CONFIGS) --analysis-days "$(ANALYSIS_DAYS)"

benchmark-replay:
	@test -n "$(MODEL_REFS)" || (echo "MODEL_REFS is required. Example: make benchmark-replay MODEL_REFS='run_a run_b'"; exit 1)
	$(RUN) scripts/benchmark_replay_models.py \
		--historical-csv "$(HISTORICAL_CSV)" \
		--weather-csv "$(WEATHER_CSV)" \
		--holidays-xlsx "$(HOLIDAYS_XLSX)" \
		--start-date "$(START_DATE)" \
		--end-date "$(END_DATE)" \
		$(MODEL_REFS)
