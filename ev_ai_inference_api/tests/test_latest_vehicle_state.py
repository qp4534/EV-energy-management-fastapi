from datetime import datetime, timezone

import pytest

from app.schemas.twins import TwinFrame
from app.services.twin_service import IncidentNotFound, TwinService


def frame() -> TwinFrame:
    return TwinFrame(
        vehicle_id="car-1",
        observed_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
        sequence=1,
        temperature_decic=[350] * 96,
        voltage_mv=[3800] * 96,
        state_level=[0] * 96,
        connector_temperature_decic=[350] * 3,
        connector_state_level=[0] * 3,
        hotspot_cell_index=0,
        hotspot_connector_index=0,
        ml_risk_level=None,
        physics_risk_level=0,
        final_risk_level=0,
    )


class FakeRedis:
    def __init__(self, value):
        self.value = value

    async def get_latest(self, vehicle_id):
        return self.value


@pytest.mark.asyncio
async def test_latest_reads_only_live_redis_snapshot() -> None:
    value = frame()
    service = TwinService(None, FakeRedis(value), None)  # type: ignore[arg-type]

    assert await service.latest("car-1") is value


@pytest.mark.asyncio
async def test_latest_does_not_substitute_an_old_incident() -> None:
    service = TwinService(None, FakeRedis(None), None)  # type: ignore[arg-type]

    with pytest.raises(IncidentNotFound, match="no live"):
        await service.latest("car-1")
