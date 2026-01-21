from typing import Optional

from fastapi import FastAPI

from ..config import AppSettings
from ..logging import init_logging
from .routes import health, nowcast, explain


def create_app(settings: Optional[AppSettings] = None) -> FastAPI:
    settings = settings or AppSettings()
    init_logging(settings.log_level)

    app = FastAPI(title=settings.app_name)

    app.include_router(health.router, prefix="/health", tags=["health"]) 
    app.include_router(nowcast.router, prefix="/nowcast", tags=["nowcast"]) 
    app.include_router(explain.router, prefix="/explain", tags=["explain"]) 

    # expose settings via app state
    app.state.settings = settings

    return app


if __name__ == "__main__":
    import uvicorn

    s = AppSettings()
    uvicorn.run(create_app(s), host=s.host, port=s.port)
