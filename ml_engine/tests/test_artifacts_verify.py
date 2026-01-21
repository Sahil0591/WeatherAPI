import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import joblib
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from ml_engine.artifacts.verify_artifacts import verify_artifacts


class TestVerifyArtifacts(unittest.TestCase):
    def _make_temp_artifacts(self):
        td = tempfile.TemporaryDirectory()
        d = td.name
        # Simple models
        clf = GradientBoostingClassifier(random_state=42)
        reg = GradientBoostingRegressor(random_state=42)
        # Fit minimal to enable predict_proba/predict
        X = [[0.0, 0.0], [1.0, 1.0]]
        y_clf = [0, 1]
        y_reg = [0.0, 1.0]
        clf.fit(X, y_clf)
        reg.fit(X, y_reg)
        joblib.dump(clf, os.path.join(d, "rain_clf.joblib"))
        joblib.dump(reg, os.path.join(d, "rain_reg.joblib"))
        # Metadata with required keys
        meta = {
            "model_name": "stormcast",
            "model_version": "2026-01-21",
            "created_at_utc": "2026-01-21T00:00:00Z",
            "horizons_min": [15, 30, 60, 120],
            "raw_columns_expected": ["temp_c", "humidity", "pressure_hpa", "wind_ms", "precip_mm"],
            "feature_columns": ["f1", "f2"],
            "training_period": {"start": "2025-01-01T00:00:00Z", "end": "2025-02-01T00:00:00Z"},
            "target_definition": "rain = precip_mm > 0.0",
        }
        with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return td

    def test_verify_success(self):
        td = self._make_temp_artifacts()
        buf = io.StringIO()
        with redirect_stdout(buf):
            verify_artifacts(td.name)
        out = buf.getvalue()
        self.assertIn("Classifier class:", out)
        self.assertIn("Regressor class:", out)
        td.cleanup()

    def test_missing_files(self):
        td = tempfile.TemporaryDirectory()
        with self.assertRaises(ValueError) as ctx:
            verify_artifacts(td.name)
        msg = str(ctx.exception)
        self.assertIn("Missing required artifact files", msg)
        td.cleanup()

    def test_missing_meta_keys(self):
        td = tempfile.TemporaryDirectory()
        # create only files and incomplete metadata
        clf = GradientBoostingClassifier(random_state=42)
        reg = GradientBoostingRegressor(random_state=42)
        X = [[0.0, 0.0], [1.0, 1.0]]
        y_clf = [0, 1]
        y_reg = [0.0, 1.0]
        clf.fit(X, y_clf)
        reg.fit(X, y_reg)
        joblib.dump(clf, os.path.join(td.name, "rain_clf.joblib"))
        joblib.dump(reg, os.path.join(td.name, "rain_reg.joblib"))
        with open(os.path.join(td.name, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"model_name": "stormcast"}, f)
        with self.assertRaises(ValueError) as ctx:
            verify_artifacts(td.name)
        msg = str(ctx.exception)
        self.assertIn("metadata.json missing required keys", msg)
        td.cleanup()


if __name__ == "__main__":
    unittest.main()
