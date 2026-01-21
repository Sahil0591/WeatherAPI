import datetime as dt
import os
import tempfile
import unittest

import pandas as pd
from sqlalchemy import create_engine

from ml_engine.ingestion.models import create_tables
from ml_engine.ingestion.storage import upsert_hourly_observations, read_recent_observations


class TestStorage(unittest.TestCase):
    def _make_engine(self):
        # Use a temporary sqlite file to avoid in-memory connection scoping issues
        td = tempfile.TemporaryDirectory()
        db_path = os.path.join(td.name, "test.db")
        url = f"sqlite:///{db_path}"
        engine = create_engine(url, future=True)
        # Attach tempdir for cleanup
        self._tmpdir = td
        self._engine = engine
        return engine

    def tearDown(self) -> None:
        # Cleanup temp directory if created
        if hasattr(self, "_engine"):
            try:
                self._engine.dispose()
            except Exception:
                pass
        if hasattr(self, "_tmpdir"):
            self._tmpdir.cleanup()

    def test_round_trip_required_columns(self) -> None:
        engine = self._make_engine()
        create_tables(engine)

        start = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
        idx = pd.date_range(start=start, periods=3, freq="h")
        df = pd.DataFrame(
            {
                "temp_c": [10.0, 11.0, 12.0],
                "humidity": [70.0, 72.0, 74.0],
                "pressure_hpa": [1015.0, 1014.5, 1014.0],
                "wind_ms": [5.0, 5.5, 6.0],
                "precip_mm": [0.0, 0.1, 0.0],
            },
            index=idx,
        )

        n = upsert_hourly_observations(engine, df, location_id=1)
        self.assertEqual(n, 3)

        out = read_recent_observations(engine, location_id=1, hours_back=48)
        self.assertEqual(len(out), 3)
        self.assertTrue(pd.api.types.is_datetime64tz_dtype(out.index))
        for col in ["temp_c", "humidity", "pressure_hpa", "wind_ms", "precip_mm"]:
            self.assertIn(col, out.columns)
        # Optional columns exist with NaN values
        self.assertIn("wind_gust_ms", out.columns)
        self.assertIn("cloud_cover", out.columns)
        self.assertTrue(pd.isna(out.iloc[0]["wind_gust_ms"]))

    def test_upsert_update(self) -> None:
        engine = self._make_engine()
        create_tables(engine)

        start = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
        base_idx = pd.date_range(start=start, periods=2, freq="h")
        df1 = pd.DataFrame(
            {
                "temp_c": [10.0, 11.0],
                "humidity": [70.0, 72.0],
                "pressure_hpa": [1015.0, 1014.5],
                "wind_ms": [5.0, 5.5],
                "precip_mm": [0.0, 0.1],
            },
            index=base_idx,
        )
        upsert_hourly_observations(engine, df1, location_id=2)

        # Update the first row's temp_c
        df2 = pd.DataFrame(
            {
                "temp_c": [15.0],
                "humidity": [70.0],
                "pressure_hpa": [1015.0],
                "wind_ms": [5.0],
                "precip_mm": [0.0],
            },
            index=base_idx[:1],
        )
        upsert_hourly_observations(engine, df2, location_id=2)

        out = read_recent_observations(engine, location_id=2, hours_back=48)
        self.assertAlmostEqual(out.iloc[0]["temp_c"], 15.0, places=6)
        self.assertAlmostEqual(out.iloc[1]["temp_c"], 11.0, places=6)


if __name__ == "__main__":
    unittest.main()
