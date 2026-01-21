from fastapi.testclient import TestClient

from weather_platform.api.main import create_app


def test_nowcast_basic():
    app = create_app()
    client = TestClient(app)

    resp = client.get("/v1/nowcast/", params={"lat": 40.7128, "lon": -74.0060, "h": [30, 60]})
    assert resp.status_code == 200
    data = resp.json()

    assert "location" in data and "lat" in data["location"] and "lon" in data["location"]
    assert data["horizons_min"] == [30, 60]
    assert "predictions" in data and isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 2

    pred = data["predictions"][0]
    assert "minutes" in pred and pred["minutes"] > 0
    assert 0.0 <= pred["p_rain"] <= 1.0
    assert pred["rain_mm_p10"] <= pred["rain_mm"] <= pred["rain_mm_p90"]
