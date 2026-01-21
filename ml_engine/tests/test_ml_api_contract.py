import datetime as dt
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from ml_engine.ml_api import StormcastModel


class TestMlApiContract(unittest.TestCase):
    def _make_temp_artifacts(self):
        td = tempfile.TemporaryDirectory()
        d = td.name
        # Tiny models trained on two features to match chosen feature_columns
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
        y_clf = np.array([0, 0, 1, 1], dtype=int)
        y_reg = np.array([0.0, 0.5, 1.0, 1.5], dtype=float)
        clf = GradientBoostingClassifier(random_state=42)
        reg = GradientBoostingRegressor(random_state=42)
        clf.fit(X, y_clf)
        reg.fit(X, y_reg)
        joblib.dump(clf, os.path.join(d, "rain_clf.joblib"))
        joblib.dump(reg, os.path.join(d, "rain_reg.joblib"))
        meta = {
            "model_name": "stormcast",
            "model_version": "2026-01-21",
            "created_at_utc": "2026-01-21T00:00:00Z",
            "horizons_min": [15, 30, 60],
            "raw_columns_expected": ["temp_c", "humidity", "pressure_hpa", "wind_ms", "precip_mm"],
            # Choose simple feature columns present in build_features output
            "feature_columns": ["hour_of_day", "month"],
            "training_period": {"start": "2025-01-01T00:00:00Z", "end": "2025-02-01T00:00:00Z"},
            "target_definition": "rain = precip_mm > 0.0",
            "metrics_summary": {"reg_rmse_raining": 0.2},
        }
        with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return td

    def _make_raw_df(self):
        start = dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        idx = pd.date_range(start=start, periods=8, freq="h")
        df = pd.DataFrame(
            {
                "temp_c": list(range(8)),
                "humidity": [10, 20, 30, 40, 50, 60, 70, 80],
                "pressure_hpa": [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007],
                "wind_ms": [1, 2, 3, 4, 5, 6, 7, 8],
                "precip_mm": [0, 0, 1, 0, 2, 0, 0, 3],
            },
            index=idx,
        )
        return df

    def test_load_predict_explain_contract(self):
        td = self._make_temp_artifacts()
        try:
            model = StormcastModel.load(td.name)
            raw_df = self._make_raw_df()
            out = model.predict(raw_df, horizons_min=[15, 30])
            self.assertIn("predictions", out)
            self.assertIn("model", out)
            self.assertIn("name", out["model"]) 
            self.assertIn("version", out["model"]) 
            preds = out["predictions"]
            self.assertTrue(isinstance(preds, list) and len(preds) == 2)
            for p in preds:
                self.assertIn("minutes", p)
                self.assertIn("p_rain", p)
                self.assertIn("rain_mm", p)
                self.assertIn("rain_mm_p10", p)
                self.assertIn("rain_mm_p90", p)
            expl = model.explain(raw_df, minutes=60)
            self.assertIn("summary", expl)
            self.assertIn("top_factors", expl)
            self.assertTrue(isinstance(expl["top_factors"], list))
        finally:
            td.cleanup()


if __name__ == "__main__":
    unittest.main()
