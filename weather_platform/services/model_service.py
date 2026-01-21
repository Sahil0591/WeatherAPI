from ..config import AppSettings


class ModelService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.model = None

    def predict(self, features: dict) -> dict:
        return {"precipitation": 0.0, "windspeed": 5.0}

    def explain(self, features: dict) -> dict:
        return {
            "top_features": [
                {"name": "temperature", "importance": 0.7},
                {"name": "humidity", "importance": 0.3},
            ]
        }
