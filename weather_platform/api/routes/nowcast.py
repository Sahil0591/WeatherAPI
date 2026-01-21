from datetime import datetime, timezone
from fastapi import APIRouter, Request

import structlog
from ...schemas.nowcast import (
    NowcastRequest,
    NowcastResponse,
    NowcastPrediction,
    Location,
    ModelInfo,
)

router = APIRouter()
logger = structlog.get_logger()


@router.post("/", response_model=NowcastResponse)
def nowcast(request: Request, payload: NowcastRequest) -> NowcastResponse:
    horizons = payload.horizons_min or [payload.horizon]

    preds = []
    for m in horizons:
        # Simple placeholder logic for demo purposes
        p_rain = round(min(1.0, (m % 100) / 100.0), 2)
        base = (abs(payload.lat) + abs(payload.lon)) % 5
        rain_mm = round(base * p_rain, 2)
        p10 = max(0.0, rain_mm - 0.5)
        p90 = rain_mm + 0.5
        preds.append(
            NowcastPrediction(
                minutes=m,
                p_rain=p_rain,
                rain_mm=rain_mm,
                rain_mm_p10=round(p10, 2),
                rain_mm_p90=round(p90, 2),
            )
        )

    resp = NowcastResponse(
        location=Location(lat=payload.lat, lon=payload.lon),
        generated_at=datetime.now(timezone.utc).isoformat(),
        horizons_min=horizons,
        predictions=preds,
        model=ModelInfo(name="stub-nowcast", version="0.1.0"),
    )

    logger.info(
        "nowcast_generated",
        lat=payload.lat,
        lon=payload.lon,
        horizons=horizons,
        predictions=len(preds),
    )

    return resp
