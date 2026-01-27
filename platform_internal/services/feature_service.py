from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ml_engine.features.build_features import build_features as _build_features

log = logging.getLogger(__name__)


class FeatureService:
    def __init__(self) -> None:
        pass

    def build_features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adapter: map normalized observation columns to ml_engine expected names and build features.

        Expects raw_df indexed by UTC datetime with columns from DataService normalization:
        - temperature_c -> temp_c
        - humidity_pct -> humidity
        - pressure_hpa -> pressure_hpa
        - windspeed_mps -> wind_ms
        - rain_mm -> precip_mm
        """
        required_map = {
            "temperature_c": "temp_c",
            "humidity_pct": "humidity",
            "pressure_hpa": "pressure_hpa",
            "windspeed_mps": "wind_ms",
            "rain_mm": "precip_mm",
        }
        missing = [src for src in required_map if src not in raw_df.columns]
        if missing:
            raise ValueError(f"Missing required normalized columns: {missing}")

        df = pd.DataFrame(index=raw_df.index)
        for src, dst in required_map.items():
            df[dst] = raw_df[src].astype(float)

        try:
            features = _build_features(df)
        except Exception as e:
            log.error("feature_build_failed", extra={"error": str(e)})
            raise
        return features
