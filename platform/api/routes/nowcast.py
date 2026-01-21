from fastapi import APIRouter, Request

import structlog
from ...schemas.nowcast import NowcastRequest, NowcastResponse
from ...services.data_service import DataService
from ...services.feature_service import FeatureService
from ...services.model_service import ModelService

router = APIRouter()
logger = structlog.get_logger()


@router.post("/", response_model=NowcastResponse)
def nowcast(request: Request, payload: NowcastRequest) -> NowcastResponse:
    settings = request.app.state.settings

    data_service = DataService(settings)
    feature_service = FeatureService(data_service)
    model_service = ModelService(settings)

    features = feature_service.build_features(payload)
    predictions = model_service.predict(features)

    logger.info("nowcast_generated", location=payload.location, timestamp=str(features.get("timestamp")))

    return NowcastResponse(
        location=payload.location,
        timestamp=str(features.get("timestamp")),
        **predictions,
    )
