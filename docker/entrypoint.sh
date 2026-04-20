#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

export PATH="/opt/venv/bin:/root/.local/bin:${PATH}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/venv}"

# Keep important runtime directories present on first start.
mkdir -p \
  /workspace/artifacts \
  /workspace/artifacts/runs \
  /workspace/artifacts/exports \
  /workspace/artifacts/models \
  /workspace/artifacts/replays \
  /workspace/artifacts/benchmarks \
  /workspace/artifacts/logs \
  /workspace/data/raw \
  /workspace/data/interim \
  /workspace/data/external

# The image is built with dependencies only. On startup we sync the mounted repo.
# With the uv cache volume, repeated runs stay fast.
if [[ -f pyproject.toml ]]; then
  if [[ -f uv.lock ]]; then
    uv sync --all-groups --frozen
  else
    uv sync --all-groups
  fi
fi

if [[ "${RUN_BOOTSTRAP_CHECKS:-1}" == "1" ]] && [[ -f Makefile ]]; then
  make doctor
fi

if [[ "${RUN_BOOTSTRAP_VERIFY:-0}" == "1" ]] && [[ -f Makefile ]]; then
  make verify
fi

exec "$@"
