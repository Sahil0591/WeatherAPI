from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _get_feature_names(X_row: Any, model: Any) -> List[str]:
    if isinstance(X_row, pd.Series):
        return [str(c) for c in X_row.index]
    if isinstance(X_row, pd.DataFrame):
        return [str(c) for c in X_row.columns]
    names = getattr(model, "feature_names_in_", None)
    if isinstance(names, (list, np.ndarray, tuple)) and len(names) > 0:
        return [str(x) for x in list(names)]
    # Fallback to positional names
    arr = np.asarray(X_row).ravel()
    return [f"f{i}" for i in range(arr.shape[0])]


def _as_row_array(X_row: Any) -> np.ndarray:
    if isinstance(X_row, pd.Series):
        return X_row.values.astype(float)
    if isinstance(X_row, pd.DataFrame):
        if len(X_row) == 1:
            return X_row.iloc[0].values.astype(float)
        raise ValueError("X_row DataFrame must have exactly one row")
    arr = np.asarray(X_row).ravel().astype(float)
    return arr


def _predict_scalar(model: Any, x_row: np.ndarray, feature_names: Optional[Sequence[str]] = None) -> float:
    # Classification probability of positive class if available; else decision_function/predict
    X_in: Any = x_row.reshape(1, -1)
    if feature_names is not None and hasattr(model, "feature_names_in_"):
        try:
            X_in = pd.DataFrame([x_row], columns=[str(c) for c in feature_names])
        except Exception:
            X_in = x_row.reshape(1, -1)
    try:
        proba = model.predict_proba(X_in)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        return float(proba[0, -1])
    except Exception:
        pass
    try:
        val = model.decision_function(X_in)
        if isinstance(val, (list, tuple, np.ndarray)):
            val = np.asarray(val).ravel()[0]
        return float(val)
    except Exception:
        pass
    # Regression fallback
    pred = model.predict(X_in)
    if isinstance(pred, (list, tuple, np.ndarray)):
        pred = np.asarray(pred).ravel()[0]
    return float(pred)


def explain_prediction(
    model: Any,
    X_row: Any,
    top_k: int = 5,
    epsilon: float = 1e-4,
) -> Dict[str, Any]:
    """Explain a single prediction, returning top feature contributions.

    If `shap` is installed, uses `shap.TreeExplainer` for tree models.
    Otherwise, falls back to global `feature_importances_` (if present) and
    local finite-difference sensitivity on the prediction scalar.

    Parameters
    ----------
    model : Any
        Fitted model (tree-based preferred). Supports scikit-learn, LightGBM, XGBoost.
    X_row : Any
        Single-row input. Supports pandas Series, single-row DataFrame, or array-like.
    top_k : int
        Number of top factors to return.
    epsilon : float
        Perturbation step for finite differences.

    Returns
    -------
    Dict[str, Any]
        {
          "summary": str,
          "top_factors": [
             {"feature": str, "direction": "up"|"down", "strength": float},
             ...
          ]
        }
    """
    names = _get_feature_names(X_row, model)
    x = _as_row_array(X_row)

    contributions: Optional[np.ndarray] = None
    used_shap = False

    # Try SHAP if available
    try:
        import shap  # type: ignore
        explainer = shap.TreeExplainer(model)  # may raise for non-tree models
        shap_values = explainer.shap_values(x.reshape(1, -1))
        if isinstance(shap_values, list):
            # Binary classification: choose positive class contributions if available
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values
        contributions = np.asarray(sv).reshape(1, -1)[0]
        used_shap = True
    except Exception:
        used_shap = False

    if contributions is None:
        # Fallback: local finite differences around X_row
        base = _predict_scalar(model, x, feature_names=names)
        contributions = np.zeros_like(x, dtype=float)
        for i in range(x.shape[0]):
            x_eps = x.copy()
            x_eps[i] = x_eps[i] + epsilon
            pred_eps = _predict_scalar(model, x_eps, feature_names=names)
            delta = pred_eps - base
            contributions[i] = delta  # absolute per-feature change for epsilon step

    # Sort by absolute impact
    order = np.argsort(-np.abs(contributions))
    top_idx = order[: top_k]

    top_factors: List[Dict[str, Any]] = []
    for i in top_idx:
        direction = "up" if contributions[i] >= 0 else "down"
        strength = float(abs(contributions[i]))
        top_factors.append({"feature": names[i], "direction": direction, "strength": strength})

    if used_shap:
        summary = "Explanation via SHAP TreeExplainer (top contributions by abs value)."
    else:
        has_importances = hasattr(model, "feature_importances_")
        summary = (
            "Explanation via local sensitivity (finite differences)"
            + (" with feature_importances_ available." if has_importances else ".")
        )

    return {"summary": summary, "top_factors": top_factors}
