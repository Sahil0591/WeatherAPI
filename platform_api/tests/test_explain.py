from fastapi.testclient import TestClient

from weather_platform.api.main import create_app


def test_explain_basic():
    app = create_app()
    client = TestClient(app)

    resp = client.post("/explain/", json={"location": "NYC"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["location"] == "NYC"
    assert "top_features" in data
    assert isinstance(data["top_features"], list)
