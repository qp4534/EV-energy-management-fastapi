from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.core.twin_redis import (
    PERSIST_STREAM,
    PREBUFFER_MAX_FRAMES,
    TwinRedisStore,
    prebuffer_key,
)
from app.db.models import INCIDENT_COMPLETE, INCIDENT_INCOMPLETE, INCIDENT_OPEN
from app.schemas.current_stage import CurrentStageResponse
from app.schemas.twins import TWIN_LAYOUT_ID, TwinFrame, TwinSampleRequest
from app.services.twin_service import (
    TwinSequenceConflict,
    TwinService,
    temperature_level,
    voltage_level,
)
from app.workers.twin_persistence import (
    IncidentProcessor,
    deterministic_incident_id,
    incident_type_for_frame,
)


START = datetime(2026, 7, 31, tzinfo=timezone.utc)


def sample(
    sequence: int = 0,
    *,
    observed_at: datetime | None = None,
    connector_peak: int = 350,
    session_id: UUID | None = None,
) -> TwinSampleRequest:
    connector = [connector_peak, 320, 300]
    return TwinSampleRequest(
        observed_at=observed_at or START + timedelta(seconds=sequence),
        session_id=session_id,
        sequence=sequence,
        temperature_decic=[350] * 96,
        voltage_mv=[3_800] * 96,
        connector_temperature_decic=connector,
        ambient_temperature_c=25.0,
        pack_current_a=90.0,
    )


def frame(sequence: int, risk: int = 0) -> TwinFrame:
    temperatures = [350] * 96
    temperatures[51] = (350, 500, 650, 850)[risk]
    states = [0] * 96
    states[51] = risk
    return TwinFrame(
        vehicle_id="car-uuid-001",
        observed_at=START + timedelta(seconds=sequence),
        sequence=sequence,
        temperature_decic=temperatures,
        voltage_mv=[3_800] * 96,
        state_level=states,
        connector_temperature_decic=[(350, 500, 650, 850)[risk], 320, 300],
        connector_state_level=[risk, 0, 0],
        hotspot_cell_index=51,
        hotspot_connector_index=0,
        ml_risk_level=None,
        physics_risk_level=risk,
        final_risk_level=risk,
    )


def test_incident_type_uses_the_dominant_visual_risk_source() -> None:
    connector = frame(0, 2).model_copy(
        update={"state_level": [0] * 96}
    )
    assert incident_type_for_frame(connector) == "connector"

    battery = frame(0, 2).model_copy(
        update={"connector_state_level": [0] * 3}
    )
    assert incident_type_for_frame(battery) == "battery"


class FakeCurrentStage:
    def __init__(self) -> None:
        self.reset_calls = []

    async def evaluate(self, vehicle_id, request):
        return CurrentStageResponse(
            vehicle_id=vehicle_id,
            sensor_health="good",
            history_seconds=1,
            model_route="warming_up",
            ml_pattern_stage=None,
            current_stage_probabilities=None,
            physical_rule_level="normal",
            final_safety_alert="normal",
            charging_equipment_observation="present_not_used_as_cell_temperature",
            reason_codes=[],
        )

    async def reset(self, vehicle_id):
        self.reset_calls.append(vehicle_id)
        return {"reset": True}


class FakeTwinRedis:
    def __init__(self) -> None:
        self.latest = None
        self.published = []
        self.events = []
        self.closed = False

    async def get_latest(self, vehicle_id):
        self.events.append("latest")
        return self.latest

    async def publish(self, value):
        self.latest = value
        self.published.append(value)

    async def subscribe(self, vehicle_id):
        self.events.append("subscribe")
        return object()

    async def live_messages(self, pubsub):
        yield self.latest
        yield frame(self.latest.sequence + 1, 1)

    async def close_subscription(self, pubsub):
        self.closed = True


@pytest.mark.asyncio
async def test_twin_contract_is_flat_and_enforces_layout_and_array_lengths() -> None:
    value = frame(0)
    assert set(value.model_dump()) == {
        "schema_version",
        "layout_id",
        "vehicle_id",
        "anomaly_id",
        "session_id",
        "observed_at",
        "sequence",
        "temperature_decic",
        "voltage_mv",
        "state_level",
        "connector_temperature_decic",
        "connector_state_level",
        "hotspot_cell_index",
        "hotspot_connector_index",
        "ml_risk_level",
            "physics_risk_level",
            "final_risk_level",
            "cell_heat_score",
            "image_risk_level",
        "image_confidence",
        "image_probabilities",
        "image_model_status",
        "module_heat_score",
        "module_state_level",
        "hotspot_module_index",
        "thermal_frame_ref",
        "thermal_frame_sha256",
        "fusion_source",
    }
    assert value.layout_id == TWIN_LAYOUT_ID
    with pytest.raises(ValidationError, match="exactly 96"):
        TwinSampleRequest(
            observed_at=START,
            sequence=0,
            temperature_decic=[350] * 95,
            voltage_mv=[3_800] * 96,
            connector_temperature_decic=[350] * 3,
        )
    with pytest.raises(ValidationError, match="literal_error"):
        TwinSampleRequest(
            layout_id="wrong-layout",
            observed_at=START,
            sequence=0,
            temperature_decic=[350] * 96,
            voltage_mv=[3_800] * 96,
            connector_temperature_decic=[350] * 3,
        )


@pytest.mark.asyncio
async def test_twin_service_enforces_exact_1hz_and_keeps_connector_visual_only() -> None:
    redis = FakeTwinRedis()
    service = TwinService(FakeCurrentStage(), redis, None)  # type: ignore[arg-type]
    first = await service.evaluate(
        "car-uuid-001", sample(connector_peak=650)
    )
    assert first.ml_risk_level is None
    assert first.connector_state_level == [2, 0, 0]
    assert first.physics_risk_level == 0
    assert first.final_risk_level == 0

    await service.evaluate("car-uuid-001", sample(1))
    with pytest.raises(TwinSequenceConflict, match="sequence"):
        await service.evaluate("car-uuid-001", sample(3))
    with pytest.raises(TwinSequenceConflict, match="one second"):
        await service.evaluate(
            "car-uuid-001",
            sample(2, observed_at=START + timedelta(seconds=3)),
        )


@pytest.mark.asyncio
async def test_new_session_resets_vehicle_history_and_allows_sequence_restart() -> None:
    redis = FakeTwinRedis()
    previous_session = UUID("11111111-1111-1111-1111-111111111111")
    next_session = UUID("22222222-2222-2222-2222-222222222222")
    redis.latest = frame(120).model_copy(update={"session_id": previous_session})
    current_stage = FakeCurrentStage()
    service = TwinService(current_stage, redis, None)  # type: ignore[arg-type]

    result = await service.evaluate(
        "car-uuid-001",
        sample(
            sequence=0,
            observed_at=START + timedelta(hours=1),
            session_id=next_session,
        ),
    )

    assert current_stage.reset_calls == ["car-uuid-001"]
    assert result.sequence == 0
    assert result.session_id == next_session


class FakeRiskCurrentStage:
    async def evaluate(self, vehicle_id, request):
        return CurrentStageResponse(
            vehicle_id=vehicle_id,
            sensor_health="good",
            history_seconds=30,
            model_route="stage_30s",
            ml_pattern_stage="caution",
            current_stage_probabilities={
                "normal": 0.1,
                "caution": 0.8,
                "warning": 0.1,
                "emergency": 0.0,
            },
            physical_rule_level="caution",
            final_safety_alert="caution",
            charging_equipment_observation="present_not_used_as_cell_temperature",
            reason_codes=["test"],
        )


class FakeAnomalyPersistence:
    def __init__(self) -> None:
        self.calls = []

    async def persist_if_anomalous(
        self, car_id, payload, inference, twin_frame=None
    ):
        self.calls.append((car_id, payload, inference, twin_frame))
        return "11111111-1111-1111-1111-111111111111"


@pytest.mark.asyncio
async def test_twin_sample_uses_observed_at_and_persists_the_same_live_frame() -> None:
    redis = FakeTwinRedis()
    persistence = FakeAnomalyPersistence()
    session_id = UUID("22222222-2222-2222-2222-222222222222")
    service = TwinService(
        FakeRiskCurrentStage(),
        redis,
        None,  # type: ignore[arg-type]
        anomaly_persistence=persistence,  # type: ignore[arg-type]
    )

    result = await service.evaluate(
        "00000000-0000-0000-0000-000000000001",
        sample(session_id=session_id),
    )

    assert result.anomaly_id == "11111111-1111-1111-1111-111111111111"
    assert redis.published == [result]
    assert len(persistence.calls) == 1
    _, model_payload, inference, twin_frame = persistence.calls[0]
    assert model_payload.timestamp_seconds == START.timestamp()
    assert model_payload.observed_at == START
    assert model_payload.session_id == session_id
    assert model_payload.temperature_decic == [350] * 96
    assert inference.final_safety_alert == "caution"
    assert twin_frame.observed_at == START


def test_cell_visual_state_uses_temperature_and_voltage_thresholds() -> None:
    assert [temperature_level(value) for value in (449, 450, 600, 800)] == [
        0,
        1,
        2,
        3,
    ]
    assert [
        voltage_level(value)
        for value in (3_800, 2_399, 2_000, 1_500, 4_351, 4_400, 4_500)
    ] == [0, 1, 2, 3, 1, 2, 3]


@pytest.mark.asyncio
async def test_live_stream_subscribes_before_latest_and_deduplicates_race() -> None:
    redis = FakeTwinRedis()
    redis.latest = frame(5)
    service = TwinService(FakeCurrentStage(), redis, None)  # type: ignore[arg-type]
    generator = service.live_frames("car-uuid-001")
    received = []
    async for item in generator:
        received.append(item.sequence)
        if len(received) == 2:
            break
    await generator.aclose()
    assert redis.events[:2] == ["subscribe", "latest"]
    assert received == [5, 6]
    assert redis.closed is True


class FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self

        return record

    async def execute(self):
        return []


class FakeRedisClient:
    def __init__(self) -> None:
        self.pipe = FakePipeline()

    def pipeline(self, transaction=True):
        assert transaction is True
        return self.pipe


@pytest.mark.asyncio
async def test_redis_write_keeps_3601_including_current_and_enqueues_persist() -> None:
    redis = FakeRedisClient()
    store = TwinRedisStore(redis)  # type: ignore[arg-type]
    await store.publish(frame(1, 1))
    xadds = [call for call in redis.pipe.calls if call[0] == "xadd"]
    assert xadds[0][1][0] == prebuffer_key("car-uuid-001")
    assert xadds[0][2]["maxlen"] == PREBUFFER_MAX_FRAMES == 3_601
    assert xadds[0][2]["approximate"] is False
    assert xadds[1][1][0] == PERSIST_STREAM


class FakeFrameStore:
    def __init__(self, frames):
        self.frames = list(frames)

    async def frames_before(self, vehicle_id, observed_at, *, limit):
        values = [item for item in self.frames if item.observed_at < observed_at]
        return values[-limit:]


class FakeRepository:
    def __init__(self) -> None:
        self.incidents = []
        self.frames = {}

    async def latest_incident(self, session, vehicle_id):
        candidates = [item for item in self.incidents if item.vehicle_id == vehicle_id]
        return candidates[-1] if candidates else None

    async def active_incidents(self, session, vehicle_id):
        return [
            item
            for item in self.incidents
            if item.vehicle_id == vehicle_id and item.status == INCIDENT_OPEN
        ]

    async def frame_is_persisted(
        self, session, *, vehicle_id, observed_at, sequence
    ):
        return any(
            value.vehicle_id == vehicle_id
            and value.observed_at == observed_at
            and value.sequence == sequence
            for target in self.frames.values()
            for value in target.values()
        )

    async def create_incident(
        self,
        session,
        *,
        incident_id,
        vehicle_id,
        triggered_at,
        window_start,
        window_end,
        incident_type="general",
    ):
        if any(item.id == incident_id for item in self.incidents):
            return
        self.incidents.append(
            SimpleNamespace(
                id=incident_id,
                vehicle_id=vehicle_id,
                triggered_at=triggered_at,
                window_start=window_start,
                window_end=window_end,
                incident_type=incident_type,
                status=INCIDENT_OPEN,
                rearmed_at=None,
            )
        )

    async def insert_frames(self, session, incident_id, values):
        target = self.frames.setdefault(incident_id, {})
        for value in values:
            target[value.observed_at] = value

    async def mark_complete(self, session, incident_id, completed_at):
        incident = next(item for item in self.incidents if item.id == incident_id)
        incident.status = INCIDENT_COMPLETE
        incident.completed_at = completed_at

    async def mark_finished(self, session, incident_id, completed_at, status):
        incident = next(item for item in self.incidents if item.id == incident_id)
        incident.status = status
        incident.completed_at = completed_at

    async def mark_rearmed(self, session, incident_id, rearmed_at):
        incident = next(item for item in self.incidents if item.id == incident_id)
        incident.rearmed_at = rearmed_at

    async def count_frames(self, session, incident_id):
        return len(self.frames.get(incident_id, {}))


@pytest.mark.asyncio
async def test_incident_worker_stores_exact_window_idempotently_and_rearms() -> None:
    pre = [frame(index, 1 if index == 3_599 else 0) for index in range(3_600)]
    trigger = frame(3_600, 1)
    store = FakeFrameStore([*pre, trigger])
    repository = FakeRepository()
    processor = IncidentProcessor(store, repository)  # type: ignore[arg-type]

    await processor.process(None, trigger)  # type: ignore[arg-type]
    incident = repository.incidents[0]
    expected_id = deterministic_incident_id("car-uuid-001", trigger.observed_at)
    assert incident.id == expected_id
    assert len(repository.frames[incident.id]) == 3_601
    assert min(repository.frames[incident.id]) == trigger.observed_at - timedelta(
        seconds=3_600
    )

    # A redelivered trigger is harmless, then 7,199 more post-trigger frames
    # complete [trigger-3600, trigger+7200) with exactly 10,800 unique rows.
    await processor.process(None, trigger)  # type: ignore[arg-type]
    for sequence in range(3_601, 10_800):
        current = frame(sequence, 1)
        store.frames.append(current)
        await processor.process(None, current)  # type: ignore[arg-type]
    assert incident.status == INCIDENT_COMPLETE
    assert len(repository.frames[incident.id]) == 10_800
    assert max(repository.frames[incident.id]) == trigger.observed_at + timedelta(
        seconds=7_199
    )

    # Sixty contiguous normal frames persist the re-arm, after which a new
    # two-second caution run may create one new deterministic incident.
    for sequence in range(10_800, 10_860):
        current = frame(sequence, 0)
        store.frames.append(current)
        await processor.process(None, current)  # type: ignore[arg-type]
    assert incident.rearmed_at == START + timedelta(seconds=10_859)
    first_risk = frame(10_860, 1)
    second_risk = frame(10_861, 1)
    store.frames.append(first_risk)
    await processor.process(None, first_risk)  # type: ignore[arg-type]
    assert len(repository.incidents) == 1
    store.frames.append(second_risk)
    await processor.process(None, second_risk)  # type: ignore[arg-type]
    assert len(repository.incidents) == 2
    assert repository.incidents[1].id != incident.id


@pytest.mark.asyncio
async def test_incident_starts_immediately_with_available_contiguous_suffix() -> None:
    first_risk = frame(0, 1)
    trigger = frame(1, 1)
    store = FakeFrameStore([first_risk, trigger])
    repository = FakeRepository()
    processor = IncidentProcessor(store, repository)  # type: ignore[arg-type]
    await processor.process(None, trigger)  # type: ignore[arg-type]
    assert len(repository.incidents) == 1
    incident = repository.incidents[0]
    assert incident.window_start == trigger.observed_at - timedelta(seconds=3_600)
    assert list(repository.frames[incident.id].values()) == [first_risk, trigger]


@pytest.mark.asyncio
async def test_stream_ack_also_deletes_single_group_message() -> None:
    redis = FakeRedisClient()
    store = TwinRedisStore(redis)  # type: ignore[arg-type]
    await store.acknowledge("twin-persistence", "123-0")
    names = [call[0] for call in redis.pipe.calls]
    assert names == ["xack", "xdel"]


@pytest.mark.asyncio
async def test_short_incident_closes_incomplete_and_persisted_replay_is_ignored() -> None:
    repository = FakeRepository()
    triggered = START + timedelta(seconds=3_600)
    incident = SimpleNamespace(
        id=deterministic_incident_id("car-uuid-001", triggered),
        vehicle_id="car-uuid-001",
        triggered_at=triggered,
        window_start=START,
        window_end=triggered + timedelta(seconds=7_200),
        status=INCIDENT_OPEN,
        rearmed_at=None,
    )
    repository.incidents.append(incident)
    repository.frames[incident.id] = {triggered: frame(3_600, 1)}
    store = FakeFrameStore([])
    processor = IncidentProcessor(store, repository)  # type: ignore[arg-type]

    after_window = frame(10_800, 1)
    await processor.process(None, after_window)  # type: ignore[arg-type]
    assert incident.status == INCIDENT_INCOMPLETE

    # Even with an adjacent high-risk frame still in the prebuffer, a reclaimed
    # pending risk message from before the durable re-arm point cannot create a
    # second incident.
    incident.status = INCIDENT_COMPLETE
    incident.rearmed_at = triggered + timedelta(seconds=60)
    pending_prior = frame(3_629, 1)
    pending = frame(3_630, 1)
    store.frames = [pending_prior, pending]
    await processor.process(None, pending)  # type: ignore[arg-type]
    assert repository.incidents == [incident]


@pytest.mark.asyncio
async def test_rearm_allows_overlapping_open_incident_and_fans_out_frames() -> None:
    first_risk = frame(0, 1)
    trigger = frame(1, 1)
    store = FakeFrameStore([first_risk, trigger])
    repository = FakeRepository()
    processor = IncidentProcessor(store, repository)  # type: ignore[arg-type]
    await processor.process(None, trigger)  # type: ignore[arg-type]
    first_incident = repository.incidents[0]

    for sequence in range(2, 62):
        current = frame(sequence, 0)
        store.frames.append(current)
        await processor.process(None, current)  # type: ignore[arg-type]
    assert first_incident.rearmed_at == START + timedelta(seconds=61)
    assert first_incident.status == INCIDENT_OPEN

    next_first = frame(62, 1)
    next_trigger = frame(63, 1)
    store.frames.append(next_first)
    await processor.process(None, next_first)  # type: ignore[arg-type]
    store.frames.append(next_trigger)
    await processor.process(None, next_trigger)  # type: ignore[arg-type]

    assert len(repository.incidents) == 2
    second_incident = repository.incidents[1]
    assert first_incident.status == second_incident.status == INCIDENT_OPEN
    assert next_trigger.observed_at in repository.frames[first_incident.id]
    assert next_trigger.observed_at in repository.frames[second_incident.id]

    shared = frame(64, 1)
    store.frames.append(shared)
    await processor.process(None, shared)  # type: ignore[arg-type]
    assert shared.observed_at in repository.frames[first_incident.id]
    assert shared.observed_at in repository.frames[second_incident.id]
