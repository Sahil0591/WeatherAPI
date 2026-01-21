from fastapi.testclient import TestClient

from weather_platform.api.main import create_app


def test_explain_basic():
    app = create_app()
    client = TestClient(app)

    resp = client.get("/v1/explain/", params={"lat": 40.7128, "lon": -74.0060, "minutes": 60})
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "top_factors" in data and isinstance(data["top_factors"], list)
