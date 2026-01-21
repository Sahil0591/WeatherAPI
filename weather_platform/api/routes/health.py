import time
from fastapi import APIRouter, Request

import structlog
from ...schemas.health import HealthResponse

router = APIRouter()
logger = structlog.get_logger()


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Health status",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {"status": "ok", "uptime_s": 12.34, "version": "0.1.0"}
                }
            },
        }
    },
)
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    uptime = max(0.0, time.time() - float(getattr(request.app.state, "start_time", time.time())))
    logger.info("health_check", env=settings.app_env)
    return HealthResponse(status="ok", uptime_s=uptime, version=settings.app_version)
