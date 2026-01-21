from ..schemas.nowcast import NowcastRequest
from .data_service import DataService


class FeatureService:
    def __init__(self, data_service: DataService):
        self.data_service = data_service

    def build_features(self, request: NowcastRequest):
        obs = self.data_service.get_observations(request.location)
        return {
            "temperature": obs["temperature"],
            "humidity": obs["humidity"],
            "timestamp": request.timestamp or "now",
        }
