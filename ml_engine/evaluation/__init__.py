from typing import List, Dict


def evaluate(model: Dict[str, float], X: List[Dict[str, float]], y: List[float]) -> Dict[str, float]:
    """Compute simple evaluation metrics for a baseline model.

    Uses Mean Absolute Error (MAE) against a constant-prediction model.

    Parameters
    ----------
    model : Dict[str, float]
        Model parameters containing at least a `bias` term.
    X : List[Dict[str, float]]
        Feature vectors (unused in bias-only model).
    y : List[float]
        Ground-truth targets.

    Returns
    -------
    Dict[str, float]
        A mapping of metric name to value.
    """
    preds = [model.get("bias", 0.0) for _ in X]
    mae: float = sum(abs(p - t) for p, t in zip(preds, y)) / len(y) if y else 0.0
    return {"mae": mae}
