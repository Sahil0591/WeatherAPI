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

## ML Pipeline (Import → Train → Verify → Evaluate)

Run these commands from the repository root.

- Import Meteostat data into SQLite:

```powershell
python -m ml_engine.ingestion.import_meteostat --db-url "sqlite:///stormcast.db" --start "2024-01-01" --end "2024-01-10"
```

- Train models from the DB (time-based split, past-only features):

```powershell
python -m ml_engine.training.train --db-url "sqlite:///stormcast.db" --location-ids "1,2,3,4,5" --start "2024-01-01T00:00:00Z" --end "2024-01-10T00:00:00Z" --artifacts-dir "ml_engine/artifacts"
```

- Verify artifacts and metadata:

```powershell
python -m ml_engine.artifacts.verify_artifacts --artifacts-dir "ml_engine/artifacts"
```

- Evaluate baselines on the same DB window (writes metrics + markdown report):

```powershell
python -m ml_engine.evaluation.evaluate --db-url "sqlite:///stormcast.db" --location-ids "1,2,3,4,5" --start "2024-01-01T00:00:00Z" --end "2024-01-10T00:00:00Z" --horizon 60 --md-path "docs/ml_report.md" --json-path "ml_engine/artifacts/metrics.json"
```

- Run tests:

```powershell
python -m pytest ml_engine/tests -v
```
