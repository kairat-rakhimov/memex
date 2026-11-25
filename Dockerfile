# syntax=docker/dockerfile:1
#
# Multi-stage image for the memex HTTP API (no GPU needed).
#
# Build:  docker build -t memex .
# Run:    docker run --rm -p 8000:8000 -e OPENAI_API_KEY \
#           -v "$PWD/kg:/app/kg" memex

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# All runtime deps ship as pure-Python or manylinux cp311 wheels — no
# toolchain needed, so the slim base installs them without build-essential.
COPY requirements.txt .

RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOST="0.0.0.0" \
    PORT="8000" \
    SAMPLE_MODE="false" \
    LOG_DIR="logs" \
    LOG_LEVEL="INFO"

WORKDIR /app

RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --home /app appuser

COPY --from=builder /opt/venv /opt/venv

COPY --chown=appuser:appgroup . .

RUN mkdir -p /app/logs /app/kg \
    && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["python", "-m", "memex", "serve"]
