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

# Copy the repository content into the image.
# In day-to-day development, docker-compose bind-mounts the repo over /workspace.
COPY . .

# Create the folders that must exist even on a fresh machine.
RUN mkdir -p \
    /workspace/artifacts \
    /workspace/artifacts/runs \
    /workspace/artifacts/exports \
    /workspace/artifacts/models/consumption/current \
    /workspace/artifacts/replays \
    /workspace/artifacts/benchmarks \
    /workspace/artifacts/logs \
    /workspace/data/raw \
    /workspace/data/interim \
    /workspace/data/external

# Install dependencies from the lock file when possible.
RUN if [ -f uv.lock ]; then \
        uv sync --all-groups --frozen; \
    else \
        uv sync --all-groups; \
    fi

COPY docker/entrypoint.sh /usr/local/bin/smartgrid-entrypoint.sh
RUN chmod +x /usr/local/bin/smartgrid-entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "smartgrid-entrypoint.sh"]
CMD ["bash"]
