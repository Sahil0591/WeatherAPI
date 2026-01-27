from __future__ import annotations

import time
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from platform_internal.services.settings import settings


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    uptime_s: float = Field(ge=0)
    version: str = Field(default=settings.APP_VERSION)


router = APIRouter()


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Health status",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {"status": "ok", "uptime_s": 12.34, "version": settings.APP_VERSION}
                }
            },
        }
    },
)
async def get_health(request: Request) -> HealthResponse:
    uptime = max(0.0, time.time() - float(getattr(request.app.state, "start_time", time.time())))
    return HealthResponse(status="ok", uptime_s=uptime, version=settings.APP_VERSION)
