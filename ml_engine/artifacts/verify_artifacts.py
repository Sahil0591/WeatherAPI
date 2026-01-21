from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import joblib

REQUIRED_FILES = ["rain_clf.joblib", "rain_reg.joblib", "metadata.json"]
REQUIRED_META_KEYS = [
    "model_name",
    "model_version",
    "created_at_utc",
    "horizons_min",
    "raw_columns_expected",
    "feature_columns",
    "training_period",
    "target_definition",
]


def verify_artifacts(artifacts_dir: str) -> None:
    """Verify presence and validity of model artifacts.

    Checks required files exist, validates required metadata keys, loads both models,
    and prints their class names.

    Parameters
    ----------
    artifacts_dir : str
        Directory containing `rain_clf.joblib`, `rain_reg.joblib`, and `metadata.json`.

    Raises
    ------
    ValueError
        If required files or metadata keys are missing.
    """
    missing: List[str] = []
    for fname in REQUIRED_FILES:
        path = os.path.join(artifacts_dir, fname)
        if not os.path.isfile(path):
            missing.append(fname)
    if missing:
        raise ValueError(f"Missing required artifact files: {', '.join(missing)}")

    meta_path = os.path.join(artifacts_dir, "metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta: Dict = json.load(f)

    missing_keys = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing_keys:
        raise ValueError(f"metadata.json missing required keys: {', '.join(missing_keys)}")

    clf = joblib.load(os.path.join(artifacts_dir, "rain_clf.joblib"))
    reg = joblib.load(os.path.join(artifacts_dir, "rain_reg.joblib"))

    print(f"Classifier class: {type(clf).__name__}")
    print(f"Regressor class: {type(reg).__name__}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify ML artifacts directory")
    p.add_argument("--artifacts-dir", required=True, help="Path to artifacts directory")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    verify_artifacts(args.artifacts_dir)


if __name__ == "__main__":
    main()
