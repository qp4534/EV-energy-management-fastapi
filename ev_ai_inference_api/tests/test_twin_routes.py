from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.twins import (
    IncidentListResponse,
    IncidentSummary,
    RiskVehicleItem,
    RiskVehicleListResponse,
    TwinFrame,
    TwinHistoryResponse,
)


START = datetime(2026, 7, 31, tzinfo=timezone.utc)


def frame() -> TwinFrame:
    return TwinFrame(
        vehicle_id="car-uuid-001",
        observed_at=START,
        sequence=0,
        temperature_decic=[350] * 96,
        voltage_mv=[3_800] * 96,
        state_level=[0] * 96,
        connector_temperature_decic=[500, 400, 350],
        connector_state_level=[1, 0, 0],
        hotspot_cell_index=0,
        hotspot_connector_index=0,
        ml_risk_level=None,
        physics_risk_level=1,
        final_risk_level=1,
    )


class FakeTwinService:
    async def evaluate(self, vehicle_id, payload):
        value = frame()
        return value.model_copy(
            update={
                "vehicle_id": vehicle_id,
                "observed_at": payload.observed_at,
                "sequence": payload.sequence,
            }
        )

    async def evaluate_observation(self, vehicle_id, payload, thermal_image):
        assert thermal_image
        value = await self.evaluate(vehicle_id, payload)
        return value.model_copy(
            update={
                "image_model_status": "unavailable",
                "fusion_source": "sensor-only",
            }
        )

    async def latest(self, vehicle_id):
        return frame().model_copy(update={"vehicle_id": vehicle_id})

    async def risk_vehicles(self):
        value = frame()
        return RiskVehicleListResponse(
            items=[
                RiskVehicleItem(
                    vehicle_id=value.vehicle_id,
                    observed_at=value.observed_at,
                    sequence=value.sequence,
                    final_risk_level=value.final_risk_level,
                )
            ]
        )

    async def incidents(self, vehicle_id):
        return IncidentListResponse(items=[self._incident(vehicle_id)])

    async def latest_history(self, vehicle_id, resolution_seconds):
        return TwinHistoryResponse(
            incident=self._incident(vehicle_id),
            resolution_seconds=resolution_seconds,
            frames=[frame().model_copy(update={"vehicle_id": vehicle_id})],
        )

    async def incident_history(self, vehicle_id, incident_id, resolution_seconds):
        return TwinHistoryResponse(
            incident=self._incident(vehicle_id).model_copy(update={"id": incident_id}),
            resolution_seconds=resolution_seconds,
            frames=[frame().model_copy(update={"vehicle_id": vehicle_id})],
        )

    async def live_frames(self, vehicle_id):
        yield frame().model_copy(update={"vehicle_id": vehicle_id})

    @staticmethod
    def _incident(vehicle_id):
        return IncidentSummary(
            id="00000000-0000-5000-8000-000000000001",
            vehicle_id=vehicle_id,
            incident_type="connector",
            triggered_at=START + timedelta(hours=1),
            window_start=START,
            window_end=START + timedelta(hours=3),
            status="complete",
            frame_count=10_800,
        )


def sample_payload():
    return {
        "schema_version": 1,
        "layout_id": "generic_ev_concept_96_v1",
        "observed_at": START.isoformat(),
        "sequence": 0,
        "temperature_decic": [350] * 96,
        "voltage_mv": [3_800] * 96,
        "connector_temperature_decic": [500, 400, 350],
    }


def test_twin_http_contracts_and_cors() -> None:
    with TestClient(app) as client:
        app.state.twin_service = FakeTwinService()
        app.state.twin_ready = True
        posted = client.post(
            "/api/v1/twins/vehicles/car-uuid-001/samples",
            json=sample_payload(),
        )
        assert posted.status_code == 200
        assert posted.json()["layout_id"] == "generic_ev_concept_96_v1"
        assert len(posted.json()["temperature_decic"]) == 96

        latest_measurement = client.get(
            "/api/v1/twins/vehicles/car-uuid-001/latest/measurement",
            params={"stale_after_seconds": 10},
        )
        assert latest_measurement.status_code == 200
        latest_payload = latest_measurement.json()
        assert latest_payload["source"] == "twin_live"
        assert latest_payload["max_cell_temperature_c"] == 35.0
        assert latest_payload["mean_cell_temperature_c"] == 35.0
        assert latest_payload["max_connector_temperature_c"] == 50.0
        assert latest_payload["min_cell_voltage_v"] == 3.8
        assert latest_payload["max_cell_voltage_v"] == 3.8
        assert latest_payload["is_stale"] is True

        risks = client.get("/api/v1/twins/risk-vehicles").json()
        assert risks == {
            "items": [
                {
                    "vehicle_id": "car-uuid-001",
                    "observed_at": "2026-07-31T00:00:00Z",
                    "sequence": 0,
                    "final_risk_level": 1,
                }
            ]
        }
        incidents = client.get(
            "/api/v1/twins/vehicles/car-uuid-001/incidents"
        ).json()
        assert incidents["items"][0]["frame_count"] == 10_800
        assert incidents["items"][0]["incident_type"] == "connector"
        history = client.get(
            "/api/v1/twins/vehicles/car-uuid-001/incidents/latest/history",
            params={"resolution_seconds": 30},
        ).json()
        assert history["resolution_seconds"] == 30
        assert len(history["frames"]) == 1
        selected_history = client.get(
            "/api/v1/twins/vehicles/car-uuid-001/incidents/"
            "00000000-0000-5000-8000-000000000001/history",
            params={"resolution_seconds": 30},
        ).json()
        assert selected_history["incident"]["incident_type"] == "connector"
        assert len(selected_history["frames"]) == 1

        preflight = client.options(
            "/api/v1/twins/risk-vehicles",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == (
            "http://localhost:5173"
        )


def test_twin_websocket_sends_public_frame() -> None:
    with TestClient(app) as client:
        app.state.twin_service = FakeTwinService()
        app.state.twin_ready = True
        with client.websocket_connect(
            "/api/v1/twins/vehicles/car-uuid-001/live"
        ) as websocket:
            payload = websocket.receive_json()
        assert payload["vehicle_id"] == "car-uuid-001"
        assert payload["ml_risk_level"] is None
        assert len(payload["connector_state_level"]) == 3


def test_twin_observation_accepts_synchronized_thermal_upload() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (255, 0, 0)).save(buffer, format="PNG")
    with TestClient(app) as client:
        app.state.twin_service = FakeTwinService()
        app.state.twin_ready = True
        response = client.post(
            "/api/v1/twins/vehicles/car-uuid-001/observations",
            data={"sample_json": __import__("json").dumps(sample_payload())},
            files={"thermal_image": ("frame.png", buffer.getvalue(), "image/png")},
        )
    assert response.status_code == 200
    assert response.json()["fusion_source"] == "sensor-only"
