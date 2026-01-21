from fastapi import APIRouter, Request

import structlog
from ...schemas.health import HealthResponse

router = APIRouter()
logger = structlog.get_logger()


@router.get("/", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    logger.info("health_check", app_env=settings.app_env)
    return HealthResponse(status="ok", app_env=settings.app_env, app_name=settings.app_name)
