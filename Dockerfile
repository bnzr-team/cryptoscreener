# Multi-stage Dockerfile for CryptoScreener-X
# DEC-029: Deployment Readiness MVP

# Stage 1: build
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
RUN pip install --no-cache-dir ".[dev]"

# Stage 2: runtime
FROM python:3.12-slim
RUN useradd -r -s /bin/false appuser
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /app/ /app/
USER appuser
EXPOSE 9090
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9090/healthz')" || exit 1
ENTRYPOINT ["python", "-m", "scripts.run_live"]
