from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import datetime as dt

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class WeatherClient(Protocol):
    """Provider-agnostic weather client interface.

    Implementations must return hourly weather observations as a pandas DataFrame
    indexed by UTC datetimes.

    Assumptions:
    - Returned index is timezone-aware (UTC).
    - Required columns always included: `temp_c`, `humidity`, `pressure_hpa`, `wind_ms`, `precip_mm`.
    - Optional columns included when available: `wind_gust_ms`, `cloud_cover`.
    - Missing values are represented as NaN. Columns may be omitted entirely if no
      data is provided by the upstream API.

    """

    def fetch_hourly(
        self,
        lat: float,
        lon: float,
        start_dt: dt.datetime,
        end_dt: dt.datetime,
    ) -> pd.DataFrame:
        """Fetch hourly weather observations.

        Parameters
        ----------
        lat : float
            Latitude in decimal degrees.
        lon : float
            Longitude in decimal degrees.
        start_dt : datetime
            Inclusive start datetime (interpreted in UTC).
        end_dt : datetime
            Exclusive end datetime (interpreted in UTC).

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by UTC datetime with columns:
            - temp_c (Celsius)
            - humidity (percent 0-100)
            - pressure_hpa (hectopascals)
            - wind_ms (meters per second)
            - precip_mm (millimeters)
            - wind_gust_ms (optional, m/s)
            - cloud_cover (optional, percent)
        """
        ...


@dataclass
class OpenMeteoClient:
    """Open-Meteo implementation of `WeatherClient` using REST API.

    Notes and assumptions:
    - Uses the hourly endpoint with timezone set to UTC.
    - Wind speed and gusts from Open-Meteo are in km/h by default; converted to m/s.
    - Missing fields are handled gracefully: absent arrays lead to missing columns
      or NaN values.
    - Retries are applied for transient HTTP errors (429/5xx) with exponential backoff.
    """

    base_url: str = "https://api.open-meteo.com/v1/forecast"
    timeout_connect: float = 5.0
    timeout_read: float = 15.0
    max_retries: int = 3
    backoff_factor: float = 0.5

    def _session(self) -> requests.Session:
        s = requests.Session()
        retries = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    def fetch_hourly(
        self,
        lat: float,
        lon: float,
        start_dt: dt.datetime,
        end_dt: dt.datetime,
    ) -> pd.DataFrame:
        # Normalize datetimes to UTC ISO-8601
        def to_utc_iso(x: dt.datetime) -> str:
            if x.tzinfo is None:
                x = x.replace(tzinfo=dt.timezone.utc)
            else:
                x = x.astimezone(dt.timezone.utc)
            return x.isoformat(timespec="seconds")

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join([
                "temperature_2m",
                "relativehumidity_2m",
                "pressure_msl",
                "windspeed_10m",
                "windgusts_10m",
                "precipitation",
                "cloudcover",
            ]),
            "timezone": "UTC",
            "start": to_utc_iso(start_dt),
            "end": to_utc_iso(end_dt),
        }

        timeout = (self.timeout_connect, self.timeout_read)
        with self._session() as s:
            resp = s.get(self.base_url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

        # Expected structure: data["hourly"]["time"], and arrays per selected field
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        # Convert times to UTC-aware pandas DatetimeIndex
        idx = pd.to_datetime(times, utc=True)

        # Helpers for extracting arrays safely
        def get_series(key: str) -> pd.Series:
            arr = hourly.get(key)
            if arr is None:
                return pd.Series([float("nan")] * len(idx), index=idx)
            # If lengths mismatch, align and pad with NaN
            arr = list(arr)
            if len(arr) < len(idx):
                arr = arr + [float("nan")] * (len(idx) - len(arr))
            elif len(arr) > len(idx):
                arr = arr[: len(idx)]
            return pd.Series(arr, index=idx)

        # Build DataFrame with required fields
        df = pd.DataFrame(index=idx)
        # Temperature (C)
        df["temp_c"] = get_series("temperature_2m").astype(float)
        # Humidity (%)
        df["humidity"] = get_series("relativehumidity_2m").astype(float)
        # Pressure (hPa)
        df["pressure_hpa"] = get_series("pressure_msl").astype(float)
        # Wind speed: km/h -> m/s
        wind_kmh = get_series("windspeed_10m").astype(float)
        df["wind_ms"] = wind_kmh * (1000.0 / 3600.0)
        # Precipitation (mm)
        df["precip_mm"] = get_series("precipitation").astype(float)

        # Optional: wind gusts (km/h -> m/s)
        gust_kmh = hourly.get("windgusts_10m")
        if gust_kmh is not None:
            gust_series = get_series("windgusts_10m").astype(float)
            df["wind_gust_ms"] = gust_series * (1000.0 / 3600.0)

        # Optional: cloud cover (%)
        cloud = hourly.get("cloudcover")
        if cloud is not None:
            df["cloud_cover"] = get_series("cloudcover").astype(float)

        # Sort index and drop duplicates if any
        df = df[~df.index.duplicated()].sort_index()
        return df
