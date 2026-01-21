from datetime import datetime, timezone
from fastapi import APIRouter, Request, HTTPException, Query

import pandas as pd
import structlog
from ...schemas.nowcast import NowcastResponse, Location

router = APIRouter()
logger = structlog.get_logger()


@router.get(
    "/",
    response_model=NowcastResponse,
    summary="Nowcast precipitation",
    responses={
        200: {
            "description": "Predicted rain metrics",
            "content": {
                "application/json": {
                    "example": {
                        "location": {"lat": 40.7, "lon": -74.0},
                        "generated_at": "2026-01-21T19:00:00Z",
                        "horizons_min": [60, 120],
                        "predictions": [
                            {"minutes": 60, "p_rain": 0.21, "rain_mm": 0.3, "rain_mm_p10": 0.0, "rain_mm_p90": 0.8},
                            {"minutes": 120, "p_rain": 0.25, "rain_mm": 0.5, "rain_mm_p10": 0.1, "rain_mm_p90": 1.0},
                        ],
                        "model": {"name": "stormcast", "version": "1.0.0"},
                    }
                }
            },
        },
        503: {"description": "Service unavailable"},
    },
)
def nowcast(
    request: Request,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    horizon: int = Query(120, gt=0, description="Default horizon in minutes; use multiples via 'horizon' or use array 'h'"),
    h: list[int] = Query(None, description="Optional multiple horizons in minutes (repeat parameter)"),
) -> NowcastResponse:
    settings = request.app.state.settings
    if getattr(request.app.state, "model_load_error", None):
        raise HTTPException(status_code=503, detail=f"Model not loaded: {request.app.state.model_load_error}")

    data_service = request.app.state.data_service
    feature_service = request.app.state.feature_service
    model_service = request.app.state.model_service

    horizons = h or [horizon]
    try:
        raw_df = data_service.fetch_recent_observations(lat=lat, lon=lon, hours=72)
        # Ensure build succeeds (requirement to call FeatureService)
        _ = feature_service.build_features(raw_df)
        out = model_service.predict(raw_df, horizons)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("nowcast_failed", error=str(e))
        raise HTTPException(status_code=503, detail="Prediction service unavailable")

    preds = out.get("predictions", [])
    model_info = out.get("model", {"name": "stormcast", "version": settings.app_version})

    return NowcastResponse(
        location=Location(lat=lat, lon=lon),
        generated_at=datetime.now(timezone.utc).isoformat(),
        horizons_min=horizons,
        predictions=preds,
        model=model_info,
    )
