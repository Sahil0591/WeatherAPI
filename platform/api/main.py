from __future__ import annotations

import time
from fastapi import FastAPI
from fastapi import Request

from platform.services.settings import settings
from platform.services.model_service import ModelService
from .routes import health


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        openapi_tags=[
            {"name": "health", "description": "Service health and uptime"},
            {"name": "v1", "description": "Versioned API"},
        ],
    )

    # App state
    app.state.settings = settings
    app.state.start_time = time.time()
    app.state.model_service = ModelService(
        artifacts_dir=settings.ARTIFACTS_DIR,
        cache_ttl_s=settings.CACHE_TTL_SECONDS,
    )

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.model_service.load()

    # Routers
    app.include_router(health.router, prefix="/health", tags=["health"])  # GET /health

    return app


app = create_app()
