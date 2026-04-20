# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH}

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        make \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Only copy dependency metadata for a fast, cacheable dev image build.
# The repository itself is mounted at runtime with .:/workspace.
COPY pyproject.toml uv.lock README.md ./
COPY docker/entrypoint.sh /usr/local/bin/smartgrid-entrypoint.sh

RUN chmod +x /usr/local/bin/smartgrid-entrypoint.sh

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --all-groups --frozen --no-install-project; \
    else \
        uv sync --all-groups --no-install-project; \
    fi

ENTRYPOINT ["/usr/bin/tini", "--", "smartgrid-entrypoint.sh"]
CMD ["bash"]
