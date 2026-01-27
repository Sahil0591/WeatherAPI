from fastapi import APIRouter, Request, HTTPException, Query

import structlog
from ...schemas.nowcast import ExplainResponse

router = APIRouter()
logger = structlog.get_logger()


@router.get(
    "/",
    response_model=ExplainResponse,
    summary="Explain nowcast",
    responses={
        200: {
            "description": "Explanation summary and top factors",
            "content": {
                "application/json": {
                    "example": {
                        "summary": "Top drivers for rain probability",
                        "top_factors": [
                            {"name": "precip_lag_1h", "contribution": 0.42},
                            {"name": "humidity_mean_3h", "contribution": 0.31},
                        ],
                    }
                }
            },
        },
        503: {"description": "Service unavailable"},
    },
)
def explain(
    request: Request,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    minutes: int = Query(60, gt=0),
) -> ExplainResponse:
    data_service = request.app.state.data_service
    feature_service = request.app.state.feature_service
    model_service = request.app.state.model_service

    try:
        raw_df = data_service.fetch_recent_observations(lat=lat, lon=lon, hours=72)
        _ = feature_service.build_features(raw_df)
        out = model_service.explain(raw_df, minutes)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("explain_failed", error=str(e))
        raise HTTPException(status_code=503, detail="Explanation service unavailable")

    return ExplainResponse(summary=out.get("summary", ""), top_factors=[
        {"name": f.get("feature") or f.get("name"), "contribution": float(f.get("strength", 0.0)) if "strength" in f else float(f.get("contribution", 0.0))}
        for f in out.get("top_factors", [])
    ])
