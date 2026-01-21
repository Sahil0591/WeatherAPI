# WeatherAPI Platform

A modular FastAPI application with structured logging, environment-based configuration, and placeholder services for nowcasting and model explainability.

## Configuration

Environment variables (prefix `APP_`):
- `APP_APP_NAME`: Application name (default: `WeatherAPI`)
- `APP_APP_ENV`: Environment (default: `development`)
- `APP_LOG_LEVEL`: Log level (default: `INFO`)
- `APP_MODEL_PATH`: Path to model file (optional)
- `APP_DATA_SOURCE_URL`: External data source URL (optional)
- `APP_HOST`: Server host (default: `0.0.0.0`)
- `APP_PORT`: Server port (default: `8000`)

You can also use a `.env` file in the project root.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\python python.exe -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\uvicorn weather_platform.api.main:create_app --factory --host 0.0.0.0 --port 8000
```

Then visit:
- `http://localhost:8000/health/`
- `http://localhost:8000/nowcast/`
- `http://localhost:8000/explain/`

## Testing

```bash
.venv\Scripts\pytest -q
```

Note: The project uses a package named `weather_platform` to avoid shadowing Python's stdlib module `platform`. If you previously had a top-level folder named `platform`, remove it or avoid running `pip` from that directory to prevent import conflicts.
