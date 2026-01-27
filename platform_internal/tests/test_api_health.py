from fastapi.testclient import TestClient

from platform_internal.api.main import create_app


def test_health_returns_ok_and_version():
    app = create_app()
    client = TestClient(app)

    resp = client.get("/health/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["uptime_s"] >= 0
    assert isinstance(data["version"], str)
