import os
import sys
import importlib.util
import sysconfig

# Bind stdlib 'platform' for third-party libs
stdlib_platform_path = os.path.join(sysconfig.get_paths()["stdlib"], "platform.py")
_spec_std = importlib.util.spec_from_file_location("platform", stdlib_platform_path)
assert _spec_std and _spec_std.loader
_stdlib_platform = importlib.util.module_from_spec(_spec_std)
_spec_std.loader.exec_module(_stdlib_platform)
sys.modules["platform"] = _stdlib_platform

from fastapi.testclient import TestClient

# Map local package to 'platform' name post stdlib bind
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_pkg_init = os.path.join(REPO_ROOT, "platform", "__init__.py")
_spec_local = importlib.util.spec_from_file_location(
    "platform", _pkg_init, submodule_search_locations=[os.path.join(REPO_ROOT, "platform")]
)
_platform_local = importlib.util.module_from_spec(_spec_local)
assert _spec_local and _spec_local.loader
_spec_local.loader.exec_module(_platform_local)
sys.modules["platform"] = _platform_local

# Shim stdlib attribute used by third-party libs
setattr(_platform_local, "python_implementation", _stdlib_platform.python_implementation)

from platform.api.main import create_app


def test_request_id_header_on_health():
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert "x-request-id" in r.headers and r.headers["x-request-id"]


def test_structured_error_on_nowcast_failure():
    app = create_app()
    # force data service to raise
    def _fail(*args, **kwargs):
        raise RuntimeError("fetch failed")
    app.state.data_service.fetch_recent_observations = _fail
    # ensure model loads stub
    os.environ["APP_ARTIFACTS_DIR"] = os.path.join(os.path.dirname(__file__), "nonexistent")
    app.state.model_service.load()

    client = TestClient(app)
    r = client.get("/v1/nowcast", params={"lat": 0, "lon": 0, "horizon": 60})
    assert r.status_code == 503
    body = r.json()
    assert "error" in body
    err = body["error"]
    assert set(err.keys()) == {"code", "message", "request_id"}
    assert err["code"] == "service_unavailable"
    assert err["message"] == "fetch failed"
    assert r.headers.get("x-request-id") == err["request_id"]
