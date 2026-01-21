import datetime as dt
import json
import unittest
from unittest.mock import patch, Mock

import pandas as pd

from ml_engine.ingestion import OpenMeteoClient


class TestOpenMeteoClient(unittest.TestCase):
    def _fake_response(self, payload: dict, status_code: int = 200) -> Mock:
        m = Mock()
        m.status_code = status_code
        m.json.return_value = payload
        m.raise_for_status.side_effect = None if status_code == 200 else Exception("HTTP error")
        return m

    @patch("requests.Session.get")
    def test_fetch_hourly_happy_path(self, mock_get: Mock) -> None:
        # Prepare fake hourly response
        times = [
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T01:00:00+00:00",
            "2024-01-01T02:00:00+00:00",
        ]
        payload = {
            "hourly": {
                "time": times,
                "temperature_2m": [10.0, 11.0, 12.0],
                "relativehumidity_2m": [80, 82, 84],
                "pressure_msl": [1015.0, 1014.5, 1014.0],
                "windspeed_10m": [18.0, 20.0, 22.0],  # km/h
                "windgusts_10m": [30.0, 33.0, 35.0],   # km/h
                "precipitation": [0.0, 0.2, 0.0],
                "cloudcover": [50, 55, 60],
            }
        }
        mock_get.return_value = self._fake_response(payload)

        client = OpenMeteoClient()
        start = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        end = dt.datetime(2024, 1, 1, 3, 0, 0, tzinfo=dt.timezone.utc)
        df = client.fetch_hourly(52.52, 13.405, start, end)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(pd.api.types.is_datetime64tz_dtype(df.index))
        self.assertListEqual(list(df.columns), [
            "temp_c",
            "humidity",
            "pressure_hpa",
            "wind_ms",
            "precip_mm",
            "wind_gust_ms",
            "cloud_cover",
        ])
        # Check conversions: 18 km/h -> 5 m/s approx (4.999...)
        self.assertAlmostEqual(df.iloc[0]["wind_ms"], 18.0 * (1000.0/3600.0), places=6)
        self.assertAlmostEqual(df.iloc[1]["wind_gust_ms"], 33.0 * (1000.0/3600.0), places=6)

    @patch("requests.Session.get")
    def test_fetch_hourly_missing_fields(self, mock_get: Mock) -> None:
        # Missing optional fields and shorter arrays
        times = [
            "2024-01-01T00:00:00+00:00",
            "2024-01-01T01:00:00+00:00",
            "2024-01-01T02:00:00+00:00",
        ]
        payload = {
            "hourly": {
                "time": times,
                "temperature_2m": [10.0, 11.0],  # shorter
                "relativehumidity_2m": [80, 82, 84],
                "pressure_msl": [1015.0, 1014.5, 1014.0],
                "windspeed_10m": [18.0, 20.0, 22.0],  # km/h
                # windgusts_10m missing
                "precipitation": [0.0, 0.2, 0.0],
                # cloudcover missing
            }
        }
        mock_get.return_value = self._fake_response(payload)

        client = OpenMeteoClient(max_retries=0)
        start = dt.datetime(2024, 1, 1, 0, 0, 0)
        end = dt.datetime(2024, 1, 1, 3, 0, 0)
        df = client.fetch_hourly(52.52, 13.405, start, end)

        # Required columns must exist
        for col in ["temp_c", "humidity", "pressure_hpa", "wind_ms", "precip_mm"]:
            self.assertIn(col, df.columns)
        # Optional columns may be absent
        self.assertNotIn("wind_gust_ms", df.columns)
        self.assertNotIn("cloud_cover", df.columns)
        # Padding with NaN for shorter arrays
        self.assertTrue(pd.isna(df.iloc[2]["temp_c"]))


if __name__ == "__main__":
    unittest.main()
