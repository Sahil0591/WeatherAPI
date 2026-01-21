from typing import List, Dict


def train(X: List[Dict[str, float]], y: List[float]) -> Dict[str, float]:
    """Train a simple baseline model.

    This stub returns a constant predictor using the mean of `y`.

    Parameters
    ----------
    X : List[Dict[str, float]]
        Feature vectors.
    y : List[float]
        Target values.

    Returns
    -------
    Dict[str, float]
        Model parameters (stub with a single bias term).
    """
    avg: float = sum(y) / len(y) if y else 0.0
    return {"bias": avg}
