from typing import Optional

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ..config import AppSettings
from ..logging import init_logging
from .routes import health, nowcast, explain
from ..services.data_service import DataService
from ..services.feature_service import FeatureService
from ..services.model_service import ModelService


def create_app(settings: Optional[AppSettings] = None) -> FastAPI:
    settings = settings or AppSettings()
    init_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load model (IO) on startup; keep other services pre-initialized for tests
        try:
            app.state.model_service.load()
        except Exception as e:
            app.state.model_load_error = str(e)
        else:
            app.state.model_load_error = None
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Service health and uptime"},
            {"name": "nowcast", "description": "Nowcast rain probability and amount"},
            {"name": "explain", "description": "Explain model predictions"},
        ],
    )

    app.include_router(health.router, prefix="/health", tags=["health"]) 
    app.include_router(nowcast.router, prefix="/v1/nowcast", tags=["nowcast"]) 
    app.include_router(explain.router, prefix="/v1/explain", tags=["explain"]) 

    app.state.settings = settings
    app.state.start_time = time.time()
    # Initialize non-IO services immediately so tests without lifespan still work
    app.state.data_service = DataService(settings)
    app.state.feature_service = FeatureService(app.state.data_service)
    app.state.model_service = ModelService(settings)
    app.state.model_load_error = None

    return app


if __name__ == "__main__":
    import uvicorn

    s = AppSettings()
    uvicorn.run(create_app(s), host=s.host, port=s.port)
