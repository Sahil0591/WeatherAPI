from fastapi.testclient import TestClient

from weather_platform.api.main import create_app


def test_health():
    app = create_app()
    client = TestClient(app)

    resp = client.get("/health/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "app_env" in data
    assert "app_name" in data
