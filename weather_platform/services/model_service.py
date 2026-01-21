from typing import Dict, Any, List

import pandas as pd

from ..config import AppSettings
from ml_engine.ml_api import StormcastModel


class ModelService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.model: StormcastModel | None = None

    def load(self) -> None:
        artifacts_dir = self.settings.model_path or "ml_engine/artifacts"
        self.model = StormcastModel.load(artifacts_dir)

    def predict(self, raw_df: pd.DataFrame, horizons_min: List[int]) -> Dict[str, Any]:
        if not self.model:
            # Fallback stub for development/testing
            preds = []
            for m in horizons_min:
                p_rain = max(0.0, min(1.0, (float(m) % 100) / 100.0))
                rain_mm = round(0.1 * (m % 10), 2)
                preds.append({
                    "minutes": int(m),
                    "p_rain": float(p_rain),
                    "rain_mm": float(rain_mm),
                    "rain_mm_p10": max(0.0, rain_mm - 0.3),
                    "rain_mm_p90": rain_mm + 0.3,
                })
            return {"predictions": preds, "model": {"name": "stub", "version": self.settings.app_version}}
        return self.model.predict(raw_df, horizons_min)

    def explain(self, raw_df: pd.DataFrame, minutes: int) -> Dict[str, Any]:
        if not self.model:
            return {
                "summary": f"Stub explanation for horizon_minutes={int(minutes)}",
                "top_factors": [
                    {"feature": "precip_lag_1h", "direction": "up", "strength": 0.4},
                    {"feature": "humidity_mean_3h", "direction": "up", "strength": 0.3},
                ],
            }
        return self.model.explain(raw_df, minutes)
