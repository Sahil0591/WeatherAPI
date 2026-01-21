from ..config import AppSettings


class DataService:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def get_observations(self, location: str):
        # Placeholder for upstream data retrieval
        return {"temperature": 20.0, "humidity": 0.5}
