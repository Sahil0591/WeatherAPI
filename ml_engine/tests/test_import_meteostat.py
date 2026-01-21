import datetime as dt
import unittest

import pandas as pd

from ml_engine.ingestion.import_meteostat import (
    meteostat_hourly_to_raw,
    extract_station_id,
)


class TestMeteostatImport(unittest.TestCase):
    def test_mapping_function(self) -> None:
        idx = pd.date_range(start=dt.datetime(2024, 1, 1, 0, 0, 0), periods=3, freq="h")
        df = pd.DataFrame(
            {
                "temp": [10.0, 11.0, 12.0],
                "rhum": [70, 72, 74],
                "pres": [1015.0, 1014.5, 1014.0],
                "prcp": [0.0, 0.1, 0.0],
                "wspd": [36.0, 18.0, 0.0],  # km/h
                "wpgt": [54.0, None, 0.0],  # km/h
            },
            index=idx,
        )
        out = meteostat_hourly_to_raw(df)
        # pandas >=2.1 deprecates is_datetime64tz_dtype
        self.assertTrue(isinstance(out.index.dtype, pd.DatetimeTZDtype))
        self.assertListEqual(
            sorted([c for c in out.columns if c in {"temp_c","humidity","pressure_hpa","precip_mm","wind_ms","wind_gust_ms"}]),
            sorted(["temp_c","humidity","pressure_hpa","precip_mm","wind_ms","wind_gust_ms"]),
        )
        self.assertAlmostEqual(out.iloc[0]["wind_ms"], 36.0/3.6, places=6)
        self.assertAlmostEqual(out.iloc[0]["wind_gust_ms"], 54.0/3.6, places=6)
        self.assertAlmostEqual(out.iloc[1]["precip_mm"], 0.1, places=6)

    def test_extract_station_id(self) -> None:
        # Case 1: id in column
        df1 = pd.DataFrame({"id": ["STN001"], "name": ["Foo"]})
        self.assertEqual(extract_station_id(df1), "STN001")

        # Case 2: id as index
        df2 = pd.DataFrame({"name": ["Bar"]}, index=pd.Index(["STN002"], name="id"))
        self.assertEqual(extract_station_id(df2), "STN002")

        # Case 3: empty
        df3 = pd.DataFrame()
        self.assertIsNone(extract_station_id(df3))


if __name__ == "__main__":
    unittest.main()
