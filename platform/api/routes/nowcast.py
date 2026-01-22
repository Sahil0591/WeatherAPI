from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


class Location(BaseModel):
    lat: float
    lon: float


class ModelInfo(BaseModel):
    name: str
    version: str


class NowcastPrediction(BaseModel):
    minutes: int
    p_rain: float
    rain_mm: float
    rain_mm_p10: float
    rain_mm_p90: float


class NowcastResponse(BaseModel):
    location: Location
    generated_at: str
    horizons_min: List[int]
    predictions: List[NowcastPrediction]
    model: ModelInfo


@router.get("/nowcast", response_model=NowcastResponse)
async def get_nowcast(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    horizon: int = Query(120, gt=0),
    horizons_min: Optional[List[int]] = Query(None),
):
    from platform.api.main import app
    settings = app.state.settings
    data_service = app.state.data_service
    model_service = app.state.model_service

    hs = horizons_min or [horizon]
    try:
        raw_df = data_service.fetch_recent_observations(lat, lon, hours_back=settings.WEATHER_HOURS_BACK)
        result = model_service.predict(raw_df, hs)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    return NowcastResponse(
        location=Location(lat=lat, lon=lon),
        generated_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        horizons_min=[int(h) for h in hs],
        predictions=[NowcastPrediction(**p) for p in result.get("predictions", [])],
        model=ModelInfo(**result.get("model", {"name": "unknown", "version": "unknown"})),
    )
