PYTHON ?= python

install:
	uv sync --all-groups

install-core:
	uv sync

install-dev:
	uv sync --group dev

install-dev-legacy:
	uv sync --all-groups

doctor:
	uv run python scripts/check_setup.py

lint:
	ruff check src tests scripts

test:
	pytest

verify:
	$(MAKE) doctor
	$(MAKE) test

train-consumption:
	$(PYTHON) scripts/train_consumption.py --promote

serve-api:
	uvicorn smartgrid.api:app --reload --host 0.0.0.0 --port 8000
