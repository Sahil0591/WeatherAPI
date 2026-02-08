from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from ml_engine.explainability.explain import explain_prediction
from ml_engine.features.build_features import build_features


def _ceil_hours(minutes: int) -> int:
    """Map minute horizons to whole-hour steps (generic API)."""
    return max(1, math.ceil(minutes / 60))


def _get_proba(model: Any, x_row: np.ndarray, feature_columns: Optional[List[str]] = None) -> float:
    """Return probability of rain for classifiers; robust fallback.

    Tries predict_proba if available, else decision_function mapped via sigmoid,
    else returns 0/1 from predict.
    """
    X_in: Any = x_row.reshape(1, -1)
    if feature_columns is not None and hasattr(model, "feature_names_in_"):
        try:
            X_in = pd.DataFrame([x_row], columns=feature_columns)
        except Exception:
            X_in = x_row.reshape(1, -1)

    try:
        proba = model.predict_proba(X_in)
        if proba.ndim == 2:
            if proba.shape[1] >= 2:
                return float(proba[0, 1])
            return float(proba[0, -1])
        return float(np.asarray(proba).ravel()[0])
    except Exception:
        pass
    try:
        val = model.decision_function(X_in)
        val = float(np.asarray(val).ravel()[0])
        return float(1.0 / (1.0 + math.exp(-val)))
    except Exception:
        pass
    try:
        pred = model.predict(X_in)
        pred = float(np.asarray(pred).ravel()[0])
        # If classifier returns 0/1, treat 1 as high probability
        return 1.0 if pred >= 1.0 else 0.0
    except Exception:
        return 0.0


@dataclass
class StormcastModel:
    """Unified API for rain probability and amount predictions.

    Methods
    -------
    load(artifacts_dir: str) -> StormcastModel
        Load artifacts from directory (classifier, regressor, metadata).
    predict(features_df: pandas.DataFrame, horizons_min: list[int]) -> dict
        Predict rain probability and amount for provided horizons (minutes).
    explain(features_df: pandas.DataFrame, minutes: int) -> dict
        Explain the most recent prediction using SHAP or sensitivity.

    Notes
    -----
    - Expects artifacts created by `ml_engine.training.train`.
    - Uses `metadata.json` for feature ordering and RMSE estimate for heuristic bounds.
    - Feature scaling used during training is not applied here; prefer tree-based models
      or extend training to persist scalers if necessary.
    """

    clf: Any
    reg: Any
    feature_columns: List[str]
    metadata: Dict[str, Any]
    model_name: str
    model_version: str

    @classmethod
    def load(cls, artifacts_dir: str) -> "StormcastModel":
        """Load artifacts and return a ready StormcastModel.

        Parameters
        ----------
        artifacts_dir : str
            Directory containing `rain_clf.joblib`, `rain_reg.joblib`, and `metadata.json`.

        Returns
        -------
        StormcastModel
        """
        clf_path = os.path.join(artifacts_dir, "rain_clf.joblib")
        reg_path = os.path.join(artifacts_dir, "rain_reg.joblib")
        meta_path = os.path.join(artifacts_dir, "metadata.json")
        if not (os.path.isfile(clf_path) and os.path.isfile(reg_path) and os.path.isfile(meta_path)):
            raise FileNotFoundError("Missing artifacts in directory: expected rain_clf.joblib, rain_reg.joblib, metadata.json")
        clf = joblib.load(clf_path)
        reg = joblib.load(reg_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        features = meta.get("feature_columns") or meta.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("metadata.json missing 'feature_columns' list")
        model_name = str(meta.get("model_name", "stormcast"))
        model_version = str(meta.get("model_version", "unknown"))
        return cls(
            clf=clf,
            reg=reg,
            feature_columns=list(features),
            metadata=meta,
            model_name=model_name,
            model_version=model_version,
        )

    def _prepare_row(self, features_df: pd.DataFrame) -> pd.Series:
        """Select and order features for the latest row, filling NaNs with zeros."""
        if features_df is None or features_df.empty:
            raise ValueError("features_df must be a non-empty DataFrame")
        missing = [c for c in self.feature_columns if c not in features_df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns for inference: {missing}")
        X = features_df.iloc[-1][self.feature_columns]
        X = X.astype(float).fillna(0.0)
        return X

    def predict(self, raw_df: pd.DataFrame, horizons_min: List[int]) -> Dict[str, Any]:
        """Predict rain probability and amounts for requested horizons.

        Parameters
        ----------
        raw_df : pandas.DataFrame
            Raw hourly DataFrame containing required columns. Features are built internally.
        horizons_min : list[int]
            Horizons in minutes to forecast. 15/30/60 map to 1h and 120 to 2h.

        Returns
        -------
        dict
            { "predictions": [ { "minutes": int, "p_rain": float, "rain_mm": float,
                                  "rain_mm_p10": float, "rain_mm_p90": float } ] }
        """
        feats_df = build_features(raw_df)
        x_row = self._prepare_row(feats_df)

        x_arr = x_row.values.astype(float)
        p = _get_proba(self.clf, x_arr, feature_columns=self.feature_columns)

        X_reg_in: Any = x_arr.reshape(1, -1)
        if hasattr(self.reg, "feature_names_in_"):
            try:
                X_reg_in = pd.DataFrame([x_arr], columns=self.feature_columns)
            except Exception:
                X_reg_in = x_arr.reshape(1, -1)
        # NOTE: the regressor is trained on "raining" rows only in ml_engine.training.train,
        # so its output is best interpreted as E[precip_mm | rain].
        amt_cond_rain = float(np.asarray(self.reg.predict(X_reg_in)).ravel()[0])

        # Heuristic bounds using RMSE estimate when available; otherwise proportional
        rmse = None
        try:
            rmse = float(self.metadata.get("metrics", {}).get("reg_rmse_raining", None))
        except Exception:
            rmse = None
        if rmse is None or not np.isfinite(rmse):
            rmse = max(0.1, abs(amt_cond_rain) * 0.5)
        z90 = 1.2815515655446004  # ~N(0,1) 90th percentile
        # Convert conditional amount bounds to unconditional expected bounds via p_rain.
        # This is an approximation but makes `rain_mm` comparable to observed precip.
        amt_expected = float(min(max(p, 0.0), 1.0)) * float(max(0.0, amt_cond_rain))
        p10 = float(min(max(p, 0.0), 1.0)) * max(0.0, amt_cond_rain - z90 * rmse)
        p90 = float(min(max(p, 0.0), 1.0)) * max(0.0, amt_cond_rain + z90 * rmse)

        preds: List[Dict[str, Any]] = []
        for minutes in horizons_min:
            steps = _ceil_hours(minutes)
            # Using single-horizon model; replicate across horizons mapping for now
            preds.append({
                "minutes": int(minutes),
                "p_rain": float(min(max(p, 0.0), 1.0)),
                "rain_mm": float(amt_expected),
                "rain_mm_p10": float(p10),
                "rain_mm_p90": float(p90),
            })
        return {"predictions": preds, "model": {"name": self.model_name, "version": self.model_version}}

    def explain(self, raw_df: pd.DataFrame, minutes: int, top_k: int = 5) -> Dict[str, Any]:
        """Explain the most recent prediction for the given horizon.

        Parameters
        ----------
        raw_df : pandas.DataFrame
            Raw hourly DataFrame; features are built internally.
        minutes : int
            Horizon in minutes (informational only; model is single-horizon).
        top_k : int
            Number of top contributing features to include.

        Returns
        -------
        dict
            { "summary": str, "top_factors": [ { "feature": str, "direction": "up"|"down",
                                                    "strength": float } ] }
        """
        # Explain classification probability by default
        feats_df = build_features(raw_df)
        x_row_series = self._prepare_row(feats_df)
        result = explain_prediction(self.clf, x_row_series, top_k=top_k)
        # Include horizon info in summary
        result["summary"] = result.get("summary", "") + f" | horizon_minutes={minutes}"
        return result
