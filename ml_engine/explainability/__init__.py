from typing import List, Dict
from .explain import explain_prediction


def explain(model: Dict[str, float], X: List[Dict[str, float]]) -> Dict[str, float]:
    """Produce simple explanation scores (stub).

    Assigns equal importance (1.0) to each observed feature name across `X`.

    Parameters
    ----------
    model : Dict[str, float]
        Model parameters (unused in this stub).
    X : List[Dict[str, float]]
        Feature vectors to inspect for feature names.

    Returns
    -------
    Dict[str, float]
        Feature importance scores.
    """
    if not X:
        return {}
    features = set().union(*[r.keys() for r in X])
    return {f: 1.0 for f in features}


__all__ = ["explain", "explain_prediction"]
