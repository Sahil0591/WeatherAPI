import unittest
import os
import tempfile

from ml_engine.ingestion import fetch_weather
from ml_engine.features import build_features_records
from ml_engine.training import train
from ml_engine.evaluation.evaluate import evaluate
from ml_engine.explainability import explain
from ml_engine.artifacts import save_model


class SmokeTests(unittest.TestCase):
    def test_pipeline_flow(self) -> None:
        data = fetch_weather("stub")
        feats = build_features_records(data)
        y = [r.get("temp", 0.0) for r in data]
        model = train(feats, y)
        metrics = evaluate(model, feats, y)
        self.assertIn("mae", metrics)
        expl = explain(model, feats)
        self.assertTrue(len(expl) > 0 or expl == {})
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.json")
            saved = save_model(model, path)
            self.assertTrue(os.path.isfile(saved))


if __name__ == "__main__":
    unittest.main()
