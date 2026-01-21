import datetime as dt
import unittest

import pandas as pd

from ml_engine.features.build_features import build_features


class TestBuildFeatures(unittest.TestCase):
    def test_feature_computation_alignment(self) -> None:
        start = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        idx = pd.date_range(start=start, periods=8, freq="h")
        df = pd.DataFrame(
            {
                "precip_mm": list(range(8)),
                "humidity": [10, 20, 30, 40, 50, 60, 70, 80],
                "pressure_hpa": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007],
                "wind_ms": [1, 2, 3, 4, 5, 6, 7, 8],
                "temp_c": list(range(8)),
            },
            index=idx,
        )
        feats = build_features(df)

        # Lags
        self.assertEqual(feats.loc[idx[3], "precip_lag_1h"], 2)
        self.assertEqual(feats.loc[idx[3], "precip_lag_3h"], 0)
        self.assertEqual(feats.loc[idx[6], "precip_lag_6h"], 0)

        # Rolling means exclude current (shifted by 1)
        # humidity_mean_3h at t3 uses t0..t2 -> mean 20
        self.assertAlmostEqual(feats.loc[idx[3], "humidity_mean_3h"], 20.0, places=6)
        # wind_mean_3h at t4 uses t1..t3 -> mean 3
        self.assertAlmostEqual(feats.loc[idx[4], "wind_mean_3h"], 3.0, places=6)
        # pressure_mean_6h at t6 uses t0..t5 -> mean 1002.5
        self.assertAlmostEqual(feats.loc[idx[6], "pressure_mean_6h"], 1002.5, places=6)

        # Deltas: current - 3h ago
        self.assertAlmostEqual(feats.loc[idx[3], "pressure_delta_3h"], 3.0, places=6)
        self.assertAlmostEqual(feats.loc[idx[3], "humidity_delta_3h"], 40 - 10, places=6)
        self.assertAlmostEqual(feats.loc[idx[3], "temp_delta_3h"], 3 - 0, places=6)

        # Std: sample std over 6 hours, shifted
        # For 1000..1005, sample std sqrt(3.5) ~= 1.870828693
        self.assertAlmostEqual(feats.loc[idx[6], "pressure_std_6h"], 3.5 ** 0.5, places=6)

        # Time features
        self.assertEqual(feats.loc[idx[0], "hour_of_day"], 0)
        self.assertEqual(feats.loc[idx[0], "day_of_week"], 0)  # Monday
        self.assertEqual(feats.loc[idx[0], "month"], 1)

    def test_resample_fills_missing_timestamps(self) -> None:
        start = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        full_idx = pd.date_range(start=start, periods=6, freq="h")
        idx_missing = full_idx.delete(2)  # remove third timestamp
        df = pd.DataFrame(
            {
                "precip_mm": [0, 1, 3, 4, 5],
                "humidity": [10, 20, 40, 50, 60],
                "pressure_hpa": [1000, 1001, 1003, 1004, 1005],
                "wind_ms": [1, 2, 4, 5, 6],
                "temp_c": [0, 1, 3, 4, 5],
            },
            index=idx_missing,
        )
        feats = build_features(df)
        # Index should include the missing timestamp
        self.assertListEqual(list(feats.index), list(full_idx))
        # The missing row should have NaNs for lagged features at appropriate times
        self.assertTrue(pd.isna(feats.loc[full_idx[3], "humidity_mean_3h"]))


if __name__ == "__main__":
    unittest.main()
