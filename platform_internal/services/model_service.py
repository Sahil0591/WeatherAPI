from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml_engine.ml_api import StormcastModel
from .cache import build_cache, Cache

log = logging.getLogger(__name__)


def _map_normalized_to_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Map normalized observation columns to ml_engine expected raw schema."""
    required_map = {
        "temperature_c": "temp_c",
        "humidity_pct": "humidity",
        "pressure_hpa": "pressure_hpa",
        "windspeed_mps": "wind_ms",
        "rain_mm": "precip_mm",
    }
    missing = [src for src in required_map if src not in df.columns]
    if missing:
        raise ValueError(f"Missing required normalized columns: {missing}")
    out = pd.DataFrame(index=pd.to_datetime(df.index, utc=True))
    for src, dst in required_map.items():
        out[dst] = df[src].astype(float)
    # sort hourly index
    out = out.sort_index()
    return out


class ModelService:
    def __init__(self, artifacts_dir: Optional[str] = None, cache_ttl_s: Optional[int] = None) -> None:
        self._model: Optional[StormcastModel] = None
        self._load_error: Optional[str] = None
        self._cache: Cache = build_cache()
        self._ttl: int = int(os.getenv("APP_CACHE_TTL_SECONDS", cache_ttl_s or 300))
        self._artifacts_dir = artifacts_dir or os.getenv("APP_ARTIFACTS_DIR", os.path.join("ml_engine", "artifacts"))

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def model_info(self) -> Dict[str, str]:
        if self._model:
            return {"name": self._model.model_name, "version": self._model.model_version}
        return {"name": "stub", "version": "dev"}

    def load(self) -> None:
        try:
            self._model = StormcastModel.load(self._artifacts_dir)
            self._load_error = None
            log.info("model_load_ok", extra={"dir": self._artifacts_dir, "name": self._model.model_name, "version": self._model.model_version})
        except Exception as e:
            self._model = None
            self._load_error = str(e)
            log.warning("model_load_failed_stub_enabled", extra={"error": str(e), "dir": self._artifacts_dir})

    def predict(self, normalized_raw_df: pd.DataFrame, horizons_min: List[int]) -> Dict[str, Any]:
        if not horizons_min:
            raise ValueError("horizons_min must be a non-empty list")
        raw_df = _map_normalized_to_raw(normalized_raw_df)
        # Cache key: last timestamp + horizons
        last_ts = pd.to_datetime(raw_df.index, utc=True).max()
        h_key = ",".join(str(int(h)) for h in horizons_min)
        key = f"nowcast:{last_ts.isoformat()}:{h_key}"
        cached = self._cache.get_json(key)
        if cached is not None:
            return cached

        if self._model is None:
            # Stub: simple heuristic based on last precip and humidity
            last_precip = float(raw_df["precip_mm"].fillna(0.0).iloc[-1])
            last_hum = float(raw_df.get("humidity", pd.Series([50.0])).fillna(50.0).iloc[-1])
            p = min(1.0, max(0.0, 0.5 * (last_precip > 0.0) + (last_hum / 200.0)))
            amt = max(0.0, last_precip)
            z90 = 1.2815515655446004
            rmse = max(0.1, amt * 0.5)
            p10 = max(0.0, amt - z90 * rmse)
            p90 = max(0.0, amt + z90 * rmse)
            preds = [{
                "minutes": int(h),
                "p_rain": float(p),
                "rain_mm": float(amt),
                "rain_mm_p10": float(p10),
                "rain_mm_p90": float(p90),
            } for h in horizons_min]
            result = {"predictions": preds, "model": self.model_info}
            self._cache.set_json(key, result, ttl_s=self._ttl)
            return result

        # Real model
        try:
            result = self._model.predict(raw_df, horizons_min)
            # Ensure numeric and types
            for p in result.get("predictions", []):
                p["minutes"] = int(p.get("minutes", 0))
                p["p_rain"] = float(np.clip(float(p.get("p_rain", 0.0)), 0.0, 1.0))
                for fld in ("rain_mm", "rain_mm_p10", "rain_mm_p90"):
                    p[fld] = float(max(0.0, float(p.get(fld, 0.0))))
            result["model"] = self.model_info
        except Exception as e:
            log.error("model_predict_failed", extra={"error": str(e)})
            raise

        self._cache.set_json(key, result, ttl_s=self._ttl)
        return result

    def explain(self, normalized_raw_df: pd.DataFrame, minutes: int) -> Dict[str, Any]:
        raw_df = _map_normalized_to_raw(normalized_raw_df)
        key = f"explain:{pd.to_datetime(raw_df.index, utc=True).max().isoformat()}:{int(minutes)}"
        cached = self._cache.get_json(key)
        if cached is not None:
            return cached

        if self._model is None:
            top = [
                {"feature": "precip_lag_1h", "direction": "up", "strength": 0.6},
                {"feature": "humidity_mean_3h", "direction": "up", "strength": 0.2},
            ]
            result = {"summary": f"stub | horizon_minutes={int(minutes)}", "top_factors": top}
            self._cache.set_json(key, result, ttl_s=self._ttl)
            return result

        try:
            result = self._model.explain(raw_df, int(minutes))
        except Exception as e:
            log.error("model_explain_failed", extra={"error": str(e)})
            raise
        self._cache.set_json(key, result, ttl_s=self._ttl)
        return result
