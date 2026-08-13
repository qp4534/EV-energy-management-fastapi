from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from fastapi.testclient import TestClient

from app.main import app
from tests.test_twin_routes import FakeTwinService, sample_payload


SERVICE_TOKEN = "service-token-with-at-least-thirty-two-characters"
TICKET_SECRET = "ticket-secret-with-at-least-thirty-two-characters"
VEHICLE_ID = "car-uuid-001"


def _ticket(vehicle_id: str, *, expires_in: int = 300) -> str:
    now = int(time.time())
    header = _segment({"alg": "HS256", "typ": "JWT"})
    payload = _segment({
        "sub": "00000000-0000-4000-8000-000000000001",
        "vehicle_id": vehicle_id,
        "scope": "twin:read",
        "iss": "ev-energy-backend",
        "aud": "ev-ai-twin",
        "iat": now,
        "exp": now + expires_in,
    })
    signing_input = f"{header}.{payload}".encode("ascii")
    key = hashlib.sha256(TICKET_SECRET.encode()).digest()
    signature = _base64url(hmac.new(key, signing_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


def _segment(value: dict) -> str:
    return _base64url(json.dumps(value, separators=(",", ":")).encode())


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _configure_auth(monkeypatch) -> None:
    monkeypatch.setenv("TWIN_AUTH_REQUIRED", "true")
    monkeypatch.setenv("TWIN_SERVICE_TOKEN", SERVICE_TOKEN)
    monkeypatch.setenv("TWIN_TICKET_SECRET", TICKET_SECRET)


def test_twin_write_requires_internal_service_token(monkeypatch) -> None:
    _configure_auth(monkeypatch)
    with TestClient(app) as client:
        app.state.twin_service = FakeTwinService()
        app.state.twin_ready = True
        denied = client.post(
            f"/api/v1/twins/vehicles/{VEHICLE_ID}/samples",
            json=sample_payload(),
        )
        accepted = client.post(
            f"/api/v1/twins/vehicles/{VEHICLE_ID}/samples",
            json=sample_payload(),
            headers={"X-Twin-Service-Token": SERVICE_TOKEN},
        )
    assert denied.status_code == 401
    assert accepted.status_code == 200


def test_twin_read_ticket_is_scoped_to_one_vehicle(monkeypatch) -> None:
    _configure_auth(monkeypatch)
    with TestClient(app) as client:
        app.state.twin_service = FakeTwinService()
        app.state.twin_ready = True
        missing = client.get(
            f"/api/v1/twins/vehicles/{VEHICLE_ID}/latest"
        )
        wrong_vehicle = client.get(
            f"/api/v1/twins/vehicles/{VEHICLE_ID}/latest",
            headers={"Authorization": f"Bearer {_ticket('another-car')}"},
        )
        accepted = client.get(
            f"/api/v1/twins/vehicles/{VEHICLE_ID}/latest",
            headers={"Authorization": f"Bearer {_ticket(VEHICLE_ID)}"},
        )
    assert missing.status_code == 401
    assert wrong_vehicle.status_code == 403
    assert accepted.status_code == 200


def test_twin_websocket_requires_ticket_subprotocol(monkeypatch) -> None:
    _configure_auth(monkeypatch)
    ticket = _ticket(VEHICLE_ID)
    with TestClient(app) as client:
        app.state.twin_service = FakeTwinService()
        app.state.twin_ready = True
        with client.websocket_connect(
            f"/api/v1/twins/vehicles/{VEHICLE_ID}/live",
            subprotocols=["twin-v1", f"auth.{ticket}"],
            headers={"Origin": "http://localhost:5173"},
        ) as websocket:
            assert websocket.accepted_subprotocol == "twin-v1"
            payload = websocket.receive_json()
    assert payload["vehicle_id"] == VEHICLE_ID
