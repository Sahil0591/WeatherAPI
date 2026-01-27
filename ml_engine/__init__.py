"""Local ML engine package for training, artifacts, and inference APIs."""
"""ML Engine package for WeatherAPI.

Subpackages:
- ingestion: Data ingestion and validation.
- features: Feature engineering and transformations.
- training: Model training routines.
- evaluation: Model evaluation metrics.
- explainability: Model interpretation utilities.
- artifacts: Model persistence and artifact handling.
- tests: Unit tests for the ml_engine package.
"""

__all__ = [
    "ingestion",
    "features",
    "training",
    "evaluation",
    "explainability",
    "artifacts",
]
