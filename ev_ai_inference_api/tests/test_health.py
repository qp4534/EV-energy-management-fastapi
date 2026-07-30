from fastapi.testclient import TestClient
from app.main import app

def test_health_and_ready_and_info():
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").status_code == 200
        info = client.get("/v1/model-info").json()
        assert info["package_version"] == "current-best-bms-hybrid-v1"
        assert info["not_a_180s_onset_model"]
