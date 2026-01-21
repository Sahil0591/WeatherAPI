import unittest
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from ml_engine.explainability.explain import explain_prediction


class TestExplainability(unittest.TestCase):
    def test_output_format(self) -> None:
        # Train a tiny model
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        y = X[:, 0] * 2.0 - X[:, 1] * 0.5 + rng.normal(scale=0.1, size=50)
        m = GradientBoostingRegressor(random_state=42)
        m.fit(X, y)

        row = pd.Series({"f0": 0.1, "f1": -0.2, "f2": 0.3})
        # Adjust model feature names for mapping
        try:
            m.feature_names_in_ = np.array(["f0", "f1", "f2"])  # type: ignore
        except Exception:
            pass

        result = explain_prediction(m, row, top_k=3)
        self.assertIn("summary", result)
        self.assertIn("top_factors", result)
        self.assertIsInstance(result["top_factors"], list)
        self.assertTrue(len(result["top_factors"]) > 0)
        f = result["top_factors"][0]
        self.assertIn("feature", f)
        self.assertIn("direction", f)
        self.assertIn("strength", f)
        self.assertIsInstance(f["feature"], str)
        self.assertIn(f["direction"], {"up", "down"})
        self.assertIsInstance(f["strength"], float)


if __name__ == "__main__":
    unittest.main()
