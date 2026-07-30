from fastapi.testclient import TestClient
import asyncio
from app.core.session_manager import SessionManager
from app.main import app
from conftest import sample

def test_vehicle_sessions_are_isolated_and_deletable():
    with TestClient(app) as client:
        for second in range(1, 31): client.post("/v1/vehicles/A/samples", json=sample(second))
        assert client.post("/v1/vehicles/B/samples", json=sample(1)).json()["history_seconds"] == 1
        assert client.delete("/v1/vehicles/A/session").json()["deleted"] is True
        assert client.delete("/v1/vehicles/A/session").status_code == 404

def test_same_vehicle_concurrent_requests_are_serialized():
    class Supervisor:
        def __init__(self): self.timestamps = []
        def reset(self): self.timestamps.clear()
        def push(self, sample, timestamp): self.timestamps.append(timestamp); return timestamp
    supervisor = Supervisor()
    manager = SessionManager(lambda: supervisor, 60, 10)
    async def exercise():
        return await asyncio.gather(manager.push("A", {}, 1), manager.push("A", {}, 2))
    assert asyncio.run(exercise()) == [1, 2]
    assert supervisor.timestamps == [1, 2]
