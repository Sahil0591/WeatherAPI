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
from fastapi.testclient import TestClient

# After pandas import, bind our local package as 'platform' for absolute imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_pkg_init = os.path.join(REPO_ROOT, "platform", "__init__.py")
_spec_local = importlib.util.spec_from_file_location(
    "platform", _pkg_init, submodule_search_locations=[os.path.join(REPO_ROOT, "platform")]
)
_platform_local = importlib.util.module_from_spec(_spec_local)
assert _spec_local and _spec_local.loader
_spec_local.loader.exec_module(_platform_local)
sys.modules["platform"] = _platform_local

# Provide stdlib shim for attributes used by third-party libs (e.g., joblib/cloudpickle)
setattr(_platform_local, "python_implementation", _stdlib_platform.python_implementation)

from platform.api.main import create_app


def _stub_df(n=6):
    times = pd.date_range("2025-01-01", periods=n, freq="H", tz="UTC")
    return pd.DataFrame(
        {
            "temperature_c": [10 + i * 0.1 for i in range(n)],
            "humidity_pct": [80 + (i % 3) for i in range(n)],
            "pressure_hpa": [1015 - i * 0.2 for i in range(n)],
            "windspeed_mps": [2.0 + (i % 4) * 0.2 for i in range(n)],
            "rain_mm": [0.0, 0.1, 0.0, 0.2, 0.0, 0.0][:n],
        },
        index=times,
    )


def test_v1_nowcast_stub():
    app = create_app()
    # stub data service
    app.state.data_service.fetch_recent_observations = lambda lat, lon, hours_back: _stub_df()
    # force stub model path
    os.environ["APP_ARTIFACTS_DIR"] = os.path.join(os.path.dirname(__file__), "nonexistent")
    app.state.model_service.load()

    client = TestClient(app)
    r = client.get("/v1/nowcast", params={"lat": 12.3, "lon": 77.6, "horizon": 60})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"location", "generated_at", "horizons_min", "predictions", "model"}
    assert body["horizons_min"] == [60]
    assert isinstance(body["predictions"], list) and len(body["predictions"]) == 1


def test_v1_explain_stub():
    app = create_app()
    app.state.data_service.fetch_recent_observations = lambda lat, lon, hours_back: _stub_df()
    os.environ["APP_ARTIFACTS_DIR"] = os.path.join(os.path.dirname(__file__), "nonexistent")
    app.state.model_service.load()

    client = TestClient(app)
    r = client.get("/v1/explain", params={"lat": 12.3, "lon": 77.6, "minutes": 60})
    assert r.status_code == 200
    body = r.json()
    assert "summary" in body and "top_factors" in body
