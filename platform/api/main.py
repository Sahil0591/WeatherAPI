from __future__ import annotations

import time
from fastapi import FastAPI

from platform.services.settings import settings
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

    # Routers
    app.include_router(health.router, prefix="/health", tags=["health"])  # GET /health

    return app


app = create_app()
