FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for building some wheels offline-friendly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip && pip install .

COPY configs ./configs
COPY scripts ./scripts
COPY domains ./domains
COPY docker_entrypoint.py ./

# Non-root user
RUN useradd -m -u 10001 ownllm && \
    mkdir -p /models /data && \
    chown -R ownllm:ownllm /app /models /data
ENV OWNLLM_DOMAINS_DIR=/app/domains
USER ownllm

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python", "docker_entrypoint.py"]
