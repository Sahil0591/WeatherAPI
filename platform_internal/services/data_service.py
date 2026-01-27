from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class OpenMeteoClient:
    base_url: str = "https://api.open-meteo.com/v1/forecast"

    def fetch_hourly(self, lat: float, lon: float, start: datetime, end: datetime) -> Dict[str, Any]:
        try:
            import requests  # type: ignore
        except Exception as e:
            raise RuntimeError(f"requests package not available: {e}") from e

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(
                [
                    "rain",
                    "precipitation",
                    "temperature_2m",
                    "relativehumidity_2m",
                    "cloudcover",
                    "windspeed_10m",
                    "winddirection_10m",
                    "pressure_msl",
                ]
            ),
            "timezone": "UTC",
            "start": start.replace(tzinfo=timezone.utc).isoformat(timespec="seconds"),
            "end": end.replace(tzinfo=timezone.utc).isoformat(timespec="seconds"),
        }

        with requests.Session() as s:
            for attempt in range(3):
                try:
                    resp = s.get(self.base_url, params=params, timeout=20)
                    resp.raise_for_status()
                    return resp.json()
                except Exception as e:
                    if attempt == 2:
                        raise
                    log.warning("openmeteo_retry", extra={"attempt": attempt + 1, "error": str(e)})


class DataService:
    def __init__(self, client: Optional[OpenMeteoClient] = None) -> None:
        self._client = client or OpenMeteoClient()

    def fetch_recent_observations(self, lat: float, lon: float, hours_back: int = 6) -> pd.DataFrame:
        if hours_back <= 0:
            raise ValueError("hours_back must be > 0")
        end = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(hours=hours_back)
        data = self._client.fetch_hourly(lat, lon, start, end)
        df = self._normalize_hourly(data)
        if df.empty:
            raise RuntimeError("no hourly observations returned")
        return df

    @staticmethod
    def _normalize_hourly(payload: Dict[str, Any]) -> pd.DataFrame:
        hourly = payload.get("hourly")
        if not isinstance(hourly, dict):
            raise ValueError("payload.hourly missing or invalid")
        times: List[str] = hourly.get("time") or []
        rain_mm: List[Optional[float]] = hourly.get("rain") or hourly.get("precipitation") or []
        temp_c: List[Optional[float]] = hourly.get("temperature_2m") or []
        humidity_pct: List[Optional[float]] = hourly.get("relativehumidity_2m") or []
        cloud_pct: List[Optional[float]] = hourly.get("cloudcover") or []
        wind_mps: List[Optional[float]] = hourly.get("windspeed_10m") or []
        wind_dir: List[Optional[float]] = hourly.get("winddirection_10m") or []
        pressure_hpa: List[Optional[float]] = hourly.get("pressure_msl") or []

        n = len(times)
        lengths = [
            len(rain_mm),
            len(temp_c),
            len(humidity_pct),
            len(cloud_pct),
            len(wind_mps),
            len(wind_dir),
            len(pressure_hpa),
        ]
        if not all(l == n for l in lengths):
            raise ValueError("hourly arrays have inconsistent lengths")

        df = pd.DataFrame(
            {
                "time": pd.to_datetime(times, utc=True),
                "rain_mm": [float(x) if x is not None else 0.0 for x in rain_mm],
                "temperature_c": [float(x) if x is not None else float("nan") for x in temp_c],
                "humidity_pct": [float(x) if x is not None else float("nan") for x in humidity_pct],
                "cloudcover_pct": [float(x) if x is not None else float("nan") for x in cloud_pct],
                "windspeed_mps": [float(x) if x is not None else float("nan") for x in wind_mps],
                "winddirection_deg": [float(x) if x is not None else float("nan") for x in wind_dir],
                "pressure_hpa": [float(x) if x is not None else float("nan") for x in pressure_hpa],
            }
        )

        df = df.sort_values("time").set_index("time")
        return df
