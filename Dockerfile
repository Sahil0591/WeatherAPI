# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps for scientific Python (wheels should cover most)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY ml_engine /app/ml_engine
COPY platform /app/platform
COPY weather_platform /app/weather_platform
COPY README.md /app/README.md

# App settings (override in deploy environment)
ENV APP_NAME="Weather Platform" \
    APP_VERSION="0.1.0" \
    APP_ARTIFACTS_DIR="/app/ml_engine/artifacts" \
    CACHE_TTL_SECONDS=300 \
    WEATHER_HOURS_BACK=6 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# Start FastAPI via uvicorn using the platform app
CMD ["python", "-m", "uvicorn", "platform.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
