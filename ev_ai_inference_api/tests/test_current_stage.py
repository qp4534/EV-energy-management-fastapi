import json
from fastapi.testclient import TestClient
from app.main import app
from conftest import sample

def test_routing_and_contracts():
    with TestClient(app) as client:
        for second in range(1, 30):
            response = client.post("/v1/vehicles/A/samples", json=sample(second)); assert response.status_code == 200
            assert response.json()["model_route"] == "warming_up"
        assert client.post("/v1/vehicles/A/samples", json=sample(30)).json()["model_route"] == "stage_30s"
        for second in range(31, 121): last = client.post("/v1/vehicles/A/samples", json=sample(second))
        assert last.json()["model_route"] == "stage_120s"
        assert abs(sum(last.json()["current_stage_probabilities"].values()) - 1) < 1e-8
        assert client.post("/v1/vehicles/A/samples", json=sample(120)).status_code == 409
        reset = client.post("/v1/vehicles/A/reset"); assert reset.status_code == 200
        assert client.post("/v1/vehicles/A/samples", json=sample(200)).json()["history_seconds"] == 1

def test_invalid_and_nonfinite_are_safe():
    with TestClient(app) as client:
        invalid = client.post("/v1/vehicles/X/samples", json=sample(1, voltage_v=7.0)).json()
        assert invalid["sensor_health"] == "invalid" and invalid["final_safety_alert"] == "unknown"
        nonfinite = json.dumps(sample(2, voltage_v=float("nan")), allow_nan=True)
        assert client.post("/v1/vehicles/X/samples", content=nonfinite, headers={"content-type": "application/json"}).status_code == 422
        assert client.post("/v1/vehicles/X/samples", json={"timestamp_seconds": 1}).status_code == 422
