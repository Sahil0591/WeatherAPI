import os
import sys
import importlib.util
import sysconfig

# 1) Force stdlib 'platform' into sys.modules before importing pandas
stdlib_platform_path = os.path.join(sysconfig.get_paths()["stdlib"], "platform.py")
spec_std = importlib.util.spec_from_file_location("platform", stdlib_platform_path)
assert spec_std and spec_std.loader
stdlib_platform = importlib.util.module_from_spec(spec_std)
spec_std.loader.exec_module(stdlib_platform)
sys.modules["platform"] = stdlib_platform

import pandas as pd

# 2) Load our local 'platform' package under an alias to avoid shadowing
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
pkg_init = os.path.join(REPO_ROOT, "platform", "__init__.py")
spec = importlib.util.spec_from_file_location(
    "platform_pkg", pkg_init, submodule_search_locations=[os.path.join(REPO_ROOT, "platform")]
)
platform_pkg = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(platform_pkg)
sys.modules["platform_pkg"] = platform_pkg

from platform_pkg.services.data_service import DataService


def test_normalize_hourly_happy_path():
    payload = {
        "hourly": {
            "time": [
                "2025-01-01T00:00:00Z",
                "2025-01-01T01:00:00Z",
                "2025-01-01T02:00:00Z",
            ],
            "rain": [0.0, 0.2, 0.0],
            "temperature_2m": [10.0, 9.5, 9.0],
            "relativehumidity_2m": [80.0, 82.0, 85.0],
            "cloudcover": [10.0, 50.0, 90.0],
            "windspeed_10m": [2.0, 3.5, 1.2],
            "winddirection_10m": [120.0, 135.0, 150.0],
            "pressure_msl": [1015.0, 1013.0, 1012.0],
        }
    }
    df = DataService._normalize_hourly(payload)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "rain_mm",
        "temperature_c",
        "humidity_pct",
        "cloudcover_pct",
        "windspeed_mps",
        "winddirection_deg",
        "pressure_hpa",
    ]
    assert len(df) == 3
    # values preserved
    assert df.iloc[1]["rain_mm"] == 0.2
    assert df.iloc[0]["temperature_c"] == 10.0


def test_normalize_inconsistent_lengths_raises():
    payload = {
        "hourly": {
            "time": ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
            "rain": [0.0],  # shorter
            "temperature_2m": [10.0, 9.5],
            "relativehumidity_2m": [80.0, 82.0],
            "cloudcover": [10.0, 50.0],
            "windspeed_10m": [2.0, 3.5],
            "winddirection_10m": [120.0, 135.0],
            "pressure_msl": [1015.0, 1013.0],
        }
    }
    try:
        DataService._normalize_hourly(payload)
        assert False, "expected ValueError"
    except ValueError:
        pass
