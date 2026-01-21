from fastapi import APIRouter, Request

import structlog
from ...schemas.explain import ExplainRequest, ExplainResponse
from ...services.data_service import DataService
from ...services.feature_service import FeatureService
from ...services.model_service import ModelService

router = APIRouter()
logger = structlog.get_logger()


@router.post("/", response_model=ExplainResponse)
def explain(request: Request, payload: ExplainRequest) -> ExplainResponse:
    settings = request.app.state.settings

    data_service = DataService(settings)
    feature_service = FeatureService(data_service)
    model_service = ModelService(settings)

    features = feature_service.build_features(payload)
    details = model_service.explain(features)

    logger.info("explain_generated", location=payload.location)

    return ExplainResponse(
        location=payload.location,
        timestamp=str(features.get("timestamp")),
        top_features=details["top_features"],
    )
