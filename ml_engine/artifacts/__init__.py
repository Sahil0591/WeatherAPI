"""Artifacts utilities: generation and verification scripts."""
from typing import Dict
import json
import os


def save_model(model: Dict[str, float], path: str) -> str:
    """Save a model to disk in JSON format.

    Parameters
    ----------
    model : Dict[str, float]
        Model parameters to persist.
    path : str
        Target file path for the JSON artifact.

    Returns
    -------
    str
        The path where the model was saved.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f)
    return path
