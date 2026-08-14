from fastapi.testclient import TestClient

from app.main import app
from conftest import sample


class CapturingPersistence:
    def __init__(self) -> None:
        self.calls = []

    async def persist_if_anomalous(self, car_id, payload, inference):
        self.calls.append((car_id, payload, inference))
        if inference.final_safety_alert == "normal":
            return None
        return "11111111-1111-1111-1111-111111111111"


def test_all_results_update_persistence_but_only_anomalies_receive_an_id():
    persistence = CapturingPersistence()
    with TestClient(app) as client:
        app.state.anomaly_persistence_enabled = True
        app.state.anomaly_persistence = persistence
        normal = client.post(
            "/v1/vehicles/00000000-0000-0000-0000-000000000001/samples",
            json=sample(1),
        )
        assert normal.status_code == 200
        assert normal.json()["anomaly_id"] is None
        assert len(persistence.calls) == 1
        assert persistence.calls[0][2].final_safety_alert == "normal"

        critical = client.post(
            "/v1/vehicles/00000000-0000-0000-0000-000000000001/samples",
            json=sample(2, raw_temp_max_c=85, raw_temp_mean_c=65),
        )
        assert critical.status_code == 200
        assert critical.json()["final_safety_alert"] == "emergency"
        assert critical.json()["anomaly_id"] == "11111111-1111-1111-1111-111111111111"
        assert len(persistence.calls) == 2
