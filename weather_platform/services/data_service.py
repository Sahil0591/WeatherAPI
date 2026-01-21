import datetime as dt
from typing import Optional

import pandas as pd

from ..config import AppSettings
from ml_engine.ingestion.client import OpenMeteoClient


class DataService:
    def __init__(self, settings: AppSettings, client: Optional[OpenMeteoClient] = None):
        self.settings = settings
        self.client = client or OpenMeteoClient()

    def fetch_recent_observations(self, lat: float, lon: float, hours: int = 72) -> pd.DataFrame:
        end_dt = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_dt = end_dt - dt.timedelta(hours=hours)
        df = self.client.fetch_hourly(lat=lat, lon=lon, start_dt=start_dt, end_dt=end_dt)
        if df is None or df.empty:
            raise RuntimeError("No observations returned from data provider")
        return df
