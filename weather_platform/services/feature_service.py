import pandas as pd

from .data_service import DataService
from ml_engine.features.build_features import build_features as _build_features


class FeatureService:
    def __init__(self, data_service: DataService):
        self.data_service = data_service

    def build_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return _build_features(raw_df)
