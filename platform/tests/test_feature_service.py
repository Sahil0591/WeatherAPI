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

from platform_pkg.services.feature_service import FeatureService


def test_feature_service_maps_columns_and_builds_features():
    # Create normalized raw observations matching DataService output
    times = pd.date_range("2025-01-01", periods=12, freq="H", tz="UTC")
    raw = pd.DataFrame(
        {
            "temperature_c": [10 + i * 0.2 for i in range(12)],
            "humidity_pct": [80 + (i % 3) for i in range(12)],
            "pressure_hpa": [1015 - i * 0.5 for i in range(12)],
            "windspeed_mps": [2 + (i % 4) * 0.3 for i in range(12)],
            "rain_mm": [0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0],
        },
        index=times,
    )

    svc = FeatureService()
    feats = svc.build_features(raw)

    # Basic assertions: DataFrame with expected feature columns and matching index length
    expected_cols = {
        "precip_lag_1h",
        "precip_lag_3h",
        "precip_lag_6h",
        "humidity_mean_3h",
        "pressure_mean_6h",
        "wind_mean_3h",
        "pressure_delta_3h",
        "humidity_delta_3h",
        "temp_delta_3h",
        "pressure_std_6h",
        "hour_of_day",
        "day_of_week",
        "month",
    }
    assert expected_cols.issubset(set(feats.columns))
    assert len(feats) == 12
