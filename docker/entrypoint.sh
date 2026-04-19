#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/venv}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:/root/.local/bin:${PATH}"

mkdir -p \
  artifacts \
  artifacts/runs \
  artifacts/exports \
  artifacts/models/consumption/current \
  artifacts/replays \
  artifacts/benchmarks \
  artifacts/logs \
  data/raw \
  data/interim \
  data/external

# Keep the environment in sync with the mounted repository.
if [[ -f pyproject.toml ]]; then
  if [[ -f uv.lock ]]; then
    uv sync --all-groups --frozen
  else
    uv sync --all-groups
  fi
fi

if [[ "${RUN_BOOTSTRAP_CHECKS:-1}" == "1" ]] && command -v make >/dev/null 2>&1; then
  make doctor
fi

if [[ "${RUN_BOOTSTRAP_VERIFY:-0}" == "1" ]] && command -v make >/dev/null 2>&1; then
  make verify
fi

exec "$@"
