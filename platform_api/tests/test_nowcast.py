from fastapi.testclient import TestClient

from weather_platform.api.main import create_app


def test_nowcast_basic():
    app = create_app()
    client = TestClient(app)

    resp = client.post("/nowcast/", json={"location": "NYC"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["location"] == "NYC"
    assert "precipitation" in data
    assert "windspeed" in data
