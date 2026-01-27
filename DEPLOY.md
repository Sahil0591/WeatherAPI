# Deploy Weather Platform (Docker + Render)

This guide covers running the FastAPI service in Docker and deploying to Render.

## Prerequisites
- Docker 24+
- Render account (optional)

## Build & Run Locally

```bash
# Build image
docker build -t weather-platform:local .

# Run container
docker run --rm -p 8000:8000 \
  -e APP_NAME="Weather Platform" \
  -e APP_VERSION="0.1.0" \
  -e APP_ARTIFACTS_DIR="/app/ml_engine/artifacts" \
  -e CACHE_TTL_SECONDS=300 \
  -e WEATHER_HOURS_BACK=6 \
  weather-platform:local

# Open API docs
# http://localhost:8000/docs
```

## Render Deployment

1. Create a new Render Web Service using the Dockerfile in this repository.
2. Set environment variables:
   - `APP_NAME` (e.g., "Weather Platform")
   - `APP_VERSION` (e.g., "0.1.0")
   - `APP_ARTIFACTS_DIR` (default `/app/ml_engine/artifacts`)
   - `CACHE_TTL_SECONDS` (e.g., `300`)
   - `WEATHER_HOURS_BACK` (e.g., `6`)
3. Expose port `8000`.
4. Health check path: `/health`.

## Notes
- Artifacts: If ML artifacts are not present, the service uses a stub fallback for predictions and explainability.
- Caching: In-memory by default; set `REDIS_URL` to enable Redis (e.g., `redis://host:6379/0`).
- Observability: Each response includes `x-request-id`. Errors follow a structured format `{ error: { code, message, request_id } }`.

## Troubleshooting
- If you see import errors related to the `platform` module on some environments, ensure you run the service via Docker as shown above. Tests use shims to avoid name collision with Python's stdlib `platform` module.
