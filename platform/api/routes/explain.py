from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


class ExplainFactor(BaseModel):
    feature: str
    direction: str
    strength: float


class ExplainResponse(BaseModel):
    summary: str
    top_factors: List[ExplainFactor]


@router.get("/explain", response_model=ExplainResponse)
async def get_explain(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    minutes: int = Query(60, gt=0),
):
    from platform.api.main import app
    settings = app.state.settings
    data_service = app.state.data_service
    model_service = app.state.model_service

    try:
        raw_df = data_service.fetch_recent_observations(lat, lon, hours_back=settings.WEATHER_HOURS_BACK)
        result = model_service.explain(raw_df, minutes)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ExplainResponse(
        summary=str(result.get("summary", "")),
        top_factors=[ExplainFactor(**f) for f in result.get("top_factors", [])],
    )
