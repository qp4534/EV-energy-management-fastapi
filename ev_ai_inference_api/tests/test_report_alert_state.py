from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from app.db.anomaly_persistence import (
    AnomalyPersistence,
    ReportAlertState,
    ReportAlertTransition,
    advance_report_alert_state,
)
from app.schemas.current_stage import CurrentStageResponse, SampleRequest


START = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
FIRST_INCIDENT = UUID("11111111-1111-4111-8111-111111111111")
SECOND_INCIDENT = UUID("22222222-2222-4222-8222-222222222222")
ANOMALY_ID = UUID("33333333-3333-4333-8333-333333333333")


def advance(
    state: ReportAlertState,
    risk_level: int,
    second: int,
    incident_id: UUID = FIRST_INCIDENT,
):
    return advance_report_alert_state(
        state,
        risk_level=risk_level,
        observed_at=START + timedelta(seconds=second),
        incident_factory=lambda: incident_id,
    )


def test_caution_requires_confirmation_and_enqueues_once() -> None:
    state = ReportAlertState()

    first = advance(state, 1, 0)
    second = advance(first.state, 1, 1)
    confirmed = advance(second.state, 1, 2)
    repeated = advance(confirmed.state, 1, 3)

    assert first.enqueue_report is False
    assert second.enqueue_report is False
    assert confirmed.enqueue_report is True
    assert confirmed.state.incident_id == FIRST_INCIDENT
    assert confirmed.state.current_risk_level == 1
    assert confirmed.state.last_reported_risk_level == 1
    assert repeated.enqueue_report is False
    assert repeated.state.incident_id == FIRST_INCIDENT


def test_escalation_enqueues_each_higher_stage_only_once() -> None:
    state = ReportAlertState(
        incident_id=FIRST_INCIDENT,
        current_risk_level=1,
        last_reported_risk_level=1,
        last_observed_at=START,
    )

    warning_1 = advance(state, 2, 1)
    warning_2 = advance(warning_1.state, 2, 2)
    warning_3 = advance(warning_2.state, 2, 3)
    emergency = advance(warning_3.state, 3, 4)
    repeated_emergency = advance(emergency.state, 3, 5)

    assert warning_1.enqueue_report is False
    assert warning_2.enqueue_report is False
    assert warning_3.enqueue_report is True
    assert warning_3.state.last_reported_risk_level == 2
    assert emergency.enqueue_report is True
    assert emergency.state.last_reported_risk_level == 3
    assert repeated_emergency.enqueue_report is False


def test_emergency_is_confirmed_immediately() -> None:
    transition = advance(ReportAlertState(), 3, 0)

    assert transition.enqueue_report is True
    assert transition.state.incident_id == FIRST_INCIDENT
    assert transition.state.current_risk_level == 3


def test_short_normal_recovery_does_not_rearm_incident() -> None:
    active = ReportAlertState(
        incident_id=FIRST_INCIDENT,
        current_risk_level=2,
        last_reported_risk_level=2,
        last_observed_at=START,
    )

    normal = advance(active, 0, 1)
    brief_recovery = advance(normal.state, 0, 9)
    emergency = advance(brief_recovery.state, 3, 10)

    assert normal.state.incident_id == FIRST_INCIDENT
    assert brief_recovery.state.incident_id == FIRST_INCIDENT
    assert emergency.state.incident_id == FIRST_INCIDENT
    assert emergency.enqueue_report is True


def test_ten_seconds_of_normal_rearms_a_new_incident() -> None:
    active = ReportAlertState(
        incident_id=FIRST_INCIDENT,
        current_risk_level=3,
        last_reported_risk_level=3,
        last_observed_at=START,
    )

    normal = advance(active, 0, 1)
    rearmed = advance(normal.state, 0, 11)
    caution_1 = advance(rearmed.state, 1, 12, SECOND_INCIDENT)
    caution_2 = advance(caution_1.state, 1, 13, SECOND_INCIDENT)
    caution_3 = advance(caution_2.state, 1, 14, SECOND_INCIDENT)

    assert rearmed.state.incident_id is None
    assert rearmed.state.last_reported_risk_level == 0
    assert caution_3.enqueue_report is True
    assert caution_3.state.incident_id == SECOND_INCIDENT


def test_risk_decrease_does_not_create_another_report() -> None:
    active = ReportAlertState(
        incident_id=FIRST_INCIDENT,
        current_risk_level=3,
        last_reported_risk_level=3,
        last_observed_at=START,
    )

    warning = advance(active, 2, 1)
    emergency_again = advance(warning.state, 3, 2)

    assert warning.enqueue_report is False
    assert emergency_again.enqueue_report is False
    assert emergency_again.state.incident_id == FIRST_INCIDENT


class FakeSession:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, dict]] = []

    async def scalar(self, statement, params):
        return ANOMALY_ID

    async def execute(self, statement, params):
        self.execute_calls.append((str(statement), params))


class FakeTransaction:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> FakeSession:
        return self.session

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class FakeSessions:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self.session)


def sample_request() -> SampleRequest:
    return SampleRequest(
        timestamp_seconds=0,
        voltage_v=3.92,
        temp_mean_c=50,
        temp_max_c=55,
        temp_delta_c=5,
        temp_saturation_fraction=0,
        temp_saturation_all=False,
        observed_at=START,
    )


def caution_response() -> CurrentStageResponse:
    return CurrentStageResponse(
        vehicle_id="00000000-0000-0000-0000-000000000001",
        sensor_health="good",
        history_seconds=30,
        model_route="stage_30s",
        ml_pattern_stage="caution",
        physical_rule_level="caution",
        final_safety_alert="caution",
        charging_equipment_observation="present_not_used_as_cell_temperature",
        reason_codes=["test"],
    )


@pytest.mark.asyncio
async def test_unconfirmed_stage_does_not_insert_report_job() -> None:
    session = FakeSession()
    persistence = AnomalyPersistence(FakeSessions(session), enqueue_report_jobs=True)

    async def unconfirmed(*args, **kwargs):
        return ReportAlertTransition(state=ReportAlertState())

    persistence._advance_report_state = unconfirmed  # type: ignore[method-assign]
    await persistence.persist_if_anomalous(
        "00000000-0000-0000-0000-000000000001",
        sample_request(),
        caution_response(),
    )

    sql = [statement for statement, _ in session.execute_calls]
    assert not any("INSERT INTO ai_report_jobs" in statement for statement in sql)


@pytest.mark.asyncio
async def test_confirmed_stage_uses_incident_and_level_as_job_key() -> None:
    session = FakeSession()
    persistence = AnomalyPersistence(FakeSessions(session), enqueue_report_jobs=True)

    async def confirmed(*args, **kwargs):
        return ReportAlertTransition(
            state=ReportAlertState(
                incident_id=FIRST_INCIDENT,
                current_risk_level=1,
                last_reported_risk_level=1,
            ),
            enqueue_report=True,
        )

    persistence._advance_report_state = confirmed  # type: ignore[method-assign]
    await persistence.persist_if_anomalous(
        "00000000-0000-0000-0000-000000000001",
        sample_request(),
        caution_response(),
    )

    report_inserts = [
        params
        for statement, params in session.execute_calls
        if "INSERT INTO ai_report_jobs" in statement
    ]
    assert len(report_inserts) == 1
    assert report_inserts[0]["job_key"] == f"ANOMALY:{FIRST_INCIDENT}:caution"
    assert report_inserts[0]["anomaly_id"] == ANOMALY_ID
