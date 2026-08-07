from __future__ import annotations

import pytest

from app.services.twin_service import (
    TwinService,
    resolve_history_vehicle_id,
)


def test_scenario_vehicle_ids_pass_through() -> None:
    assert (
        resolve_history_vehicle_id("scenario-connector_local_overheat", None)
        == "scenario-connector_local_overheat"
    )


def test_car_uuid_maps_to_anomaly_scenario() -> None:
    assert (
        resolve_history_vehicle_id(
            "11111111-1111-1111-1111-111111111111",
            "커넥터 국부 과열",
        )
        == "scenario-connector_local_overheat"
    )


def test_car_without_log_maps_to_normal() -> None:
    assert (
        resolve_history_vehicle_id(
            "11111111-1111-1111-1111-111111111111",
            None,
        )
        == "scenario-normal"
    )


class FakeRepository:
    async def latest_abnormal_type_for_car(self, session, car_id: str) -> str:
        return "셀 전압 불균형"


@pytest.mark.asyncio
async def test_service_resolves_vehicle_through_repository() -> None:
    service = TwinService.__new__(TwinService)
    service.repository = FakeRepository()
    resolved = await service._resolve_history_vehicle(
        session=None,
        vehicle_id="11111111-1111-1111-1111-111111111111",
    )
    assert resolved == "scenario-cell_voltage_imbalance"
