from typing import List, Dict


def build_features_records(records: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Transform raw record list into simple model-ready features (legacy stub).

    Parameters
    ----------
    records : List[Dict[str, float]]
        Raw input records, each represented as a mapping of feature name to value.

    Returns
    -------
    List[Dict[str, float]]
        Transformed records containing engineered features.

    Notes
    -----
    This legacy helper keeps smoke tests working for record-based inputs.
    The primary feature pipeline is in `ml_engine.features.build_features.build_features`
    which operates on pandas DataFrames.
    """
    transformed: List[Dict[str, float]] = []
    for r in records:
        out: Dict[str, float] = dict(r)
        if "humidity" in out:
            out["humidity_pct"] = out["humidity"] * 100.0
        transformed.append(out)
    return transformed


__all__ = ["build_features_records"]
