import datetime as dt
import os
import unittest

import pandas as pd

from ml_engine.evaluation.evaluate import (
    persistence_baseline,
    moving_average_baseline,
    compute_labels,
    evaluate_metrics,
    evaluate_and_save,
)


class TestEvaluation(unittest.TestCase):
    def _make_df(self) -> pd.DataFrame:
        start = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        idx = pd.date_range(start=start, periods=10, freq="h")
        return pd.DataFrame({"precip_mm": [0, 0, 1, 0, 2, 0, 0, 3, 0, 0]}, index=idx)

    def test_baselines_and_labels(self) -> None:
        df = self._make_df()
        # Persistence uses last observed value aligned to current time
        self.assertEqual(float(persistence_baseline(df).iloc[2]), 1.0)
        # Moving average uses last 3 including current; first two NaN
        ma = moving_average_baseline(df)
        self.assertTrue(pd.isna(ma.iloc[1]))
        self.assertAlmostEqual(float(ma.iloc[4]), (1+0+2)/3, places=6)
        # Labels shifted forward by 1 hour
        y_reg, y_clf = compute_labels(df, horizon_steps=1)
        self.assertEqual(float(y_reg.iloc[2]), 0.0)  # next hour after index[2]
        self.assertEqual(int(y_clf.iloc[2]), 0)

    def test_evaluate_and_save(self) -> None:
        df = self._make_df()
        metrics = evaluate_metrics(df, horizon_steps=1)
        self.assertIn("persist_mae", metrics)
        self.assertIn("ma_rmse", metrics)
        # Save report
        md_path = os.path.join("docs", "ml_report.md")
        json_path = os.path.join("ml_engine", "artifacts", "metrics.json")
        out = evaluate_and_save(df, horizon_steps=1, md_path=md_path, json_path=json_path)
        self.assertTrue(os.path.isfile(md_path))
        self.assertTrue(os.path.isfile(json_path))


if __name__ == "__main__":
    unittest.main()
