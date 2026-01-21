import os
import sys
import importlib.util
import sysconfig

# Bind stdlib 'platform' for pandas
stdlib_platform_path = os.path.join(sysconfig.get_paths()["stdlib"], "platform.py")
_spec_std = importlib.util.spec_from_file_location("platform", stdlib_platform_path)
assert _spec_std and _spec_std.loader
_stdlib_platform = importlib.util.module_from_spec(_spec_std)
_spec_std.loader.exec_module(_stdlib_platform)
sys.modules["platform"] = _stdlib_platform

import pandas as pd

# Load local 'platform' package under alias to avoid name collision
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_pkg_init = os.path.join(REPO_ROOT, "platform", "__init__.py")
_spec = importlib.util.spec_from_file_location(
    "platform_pkg", _pkg_init, submodule_search_locations=[os.path.join(REPO_ROOT, "platform")]
)
_platform_pkg = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_platform_pkg)
sys.modules["platform_pkg"] = _platform_pkg

from platform_pkg.services.model_service import ModelService


def test_model_service_stub_predict_shape():
    # Ensure no artifacts so stub path is used
    os.environ["APP_ARTIFACTS_DIR"] = os.path.join(REPO_ROOT, "nonexistent_artifacts")
    svc = ModelService()
    svc.load()

    # Normalized raw observations
    times = pd.date_range("2025-01-01", periods=6, freq="H", tz="UTC")
    raw = pd.DataFrame(
        {
            "temperature_c": [10, 10.2, 9.9, 10.1, 10.0, 9.8],
            "humidity_pct": [80, 81, 82, 83, 82, 80],
            "pressure_hpa": [1015, 1014.5, 1014, 1013.8, 1013.6, 1013.4],
            "windspeed_mps": [2.0, 2.3, 1.8, 2.1, 2.2, 2.0],
            "rain_mm": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        },
        index=times,
    )

    result = svc.predict(raw, horizons_min=[30, 60, 120])
    assert isinstance(result, dict)
    preds = result.get("predictions")
    assert isinstance(preds, list) and len(preds) == 3
    for p in preds:
        assert set(p.keys()) == {"minutes", "p_rain", "rain_mm", "rain_mm_p10", "rain_mm_p90"}
        assert 0 <= p["p_rain"] <= 1
        assert p["minutes"] in (30, 60, 120)


def test_model_service_explain_stub():
    os.environ["APP_ARTIFACTS_DIR"] = os.path.join(REPO_ROOT, "nonexistent_artifacts")
    svc = ModelService()
    svc.load()

    times = pd.date_range("2025-01-01", periods=6, freq="H", tz="UTC")
    raw = pd.DataFrame(
        {
            "temperature_c": [10]*6,
            "humidity_pct": [80]*6,
            "pressure_hpa": [1015]*6,
            "windspeed_mps": [2]*6,
            "rain_mm": [0]*6,
        },
        index=times,
    )
    result = svc.explain(raw, minutes=60)
    assert isinstance(result, dict)
    assert "summary" in result and "top_factors" in result
