from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Callable
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.schemas.current_stage import CurrentStageResponse, SampleRequest
from app.schemas.twins import TwinFrame


ALERT_RISK_TEXT = {
    "caution": "주의",
    "warning": "경고",
    "emergency": "긴급",
}
ALERT_RISK_NUMBER = {
    "normal": 0,
    "caution": 1,
    "warning": 2,
    "emergency": 3,
}
REPORT_RISK_CONFIRMATION_SAMPLES = 3
REPORT_NORMAL_REARM_SECONDS = 10


@dataclass(frozen=True)
class ReportAlertState:
    """Durable per-vehicle state used only to throttle AI report jobs."""

    incident_id: UUID | None = None
    current_risk_level: int = 0
    last_reported_risk_level: int = 0
    candidate_risk_level: int | None = None
    candidate_count: int = 0
    normal_since: datetime | None = None
    last_observed_at: datetime | None = None


@dataclass(frozen=True)
class ReportAlertTransition:
    state: ReportAlertState
    enqueue_report: bool = False


def advance_report_alert_state(
    state: ReportAlertState,
    *,
    risk_level: int,
    observed_at: datetime,
    confirmation_samples: int = REPORT_RISK_CONFIRMATION_SAMPLES,
    normal_rearm_seconds: int = REPORT_NORMAL_REARM_SECONDS,
    incident_factory: Callable[[], UUID] = uuid4,
) -> ReportAlertTransition:
    """Advance one vehicle's report state without changing inference persistence.

    Caution and warning must repeat before they open/escalate an incident. Emergency
    is immediate. A report is emitted only for a risk level higher than any level
    already reported in the active incident. Ten continuous normal seconds rearm the
    state so a later abnormal period becomes a new incident.
    """

    if risk_level not in {0, 1, 2, 3}:
        raise ValueError("risk_level must be between 0 and 3")
    if confirmation_samples <= 0:
        raise ValueError("confirmation_samples must be positive")
    if normal_rearm_seconds <= 0:
        raise ValueError("normal_rearm_seconds must be positive")
    if state.last_observed_at is not None and observed_at < state.last_observed_at:
        return ReportAlertTransition(state=state)

    if risk_level == 0:
        if state.incident_id is None:
            return ReportAlertTransition(
                state=ReportAlertState(last_observed_at=observed_at)
            )

        normal_since = state.normal_since or observed_at
        if observed_at - normal_since >= timedelta(seconds=normal_rearm_seconds):
            return ReportAlertTransition(
                state=ReportAlertState(last_observed_at=observed_at)
            )
        return ReportAlertTransition(
            state=replace(
                state,
                candidate_risk_level=None,
                candidate_count=0,
                normal_since=normal_since,
                last_observed_at=observed_at,
            )
        )

    # Any abnormal sample interrupts a possible normal-recovery window.
    base = replace(state, normal_since=None, last_observed_at=observed_at)
    if risk_level <= state.current_risk_level:
        return ReportAlertTransition(
            state=replace(
                base,
                current_risk_level=risk_level,
                candidate_risk_level=None,
                candidate_count=0,
            )
        )

    if risk_level == ALERT_RISK_NUMBER["emergency"]:
        confirmed = True
        candidate_count = 0
    else:
        candidate_count = (
            state.candidate_count + 1
            if state.candidate_risk_level == risk_level
            else 1
        )
        confirmed = candidate_count >= confirmation_samples

    if not confirmed:
        return ReportAlertTransition(
            state=replace(
                base,
                candidate_risk_level=risk_level,
                candidate_count=candidate_count,
            )
        )

    incident_id = state.incident_id or incident_factory()
    enqueue_report = risk_level > state.last_reported_risk_level
    return ReportAlertTransition(
        state=replace(
            base,
            incident_id=incident_id,
            current_risk_level=risk_level,
            last_reported_risk_level=max(
                state.last_reported_risk_level, risk_level
            ),
            candidate_risk_level=None,
            candidate_count=0,
        ),
        enqueue_report=enqueue_report,
    )


class AnomalyPersistence:
    """Writes only abnormal current-stage results to the shared ERD tables."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        enqueue_report_jobs: bool = False,
    ) -> None:
        self._sessions = sessions
        self._enqueue_report_jobs = enqueue_report_jobs

    async def persist_if_anomalous(
        self,
        car_id: str,
        payload: SampleRequest,
        inference: CurrentStageResponse,
        twin_frame: TwinFrame | None = None,
    ) -> str | None:
        alert = inference.final_safety_alert
        if alert not in ALERT_RISK_NUMBER:
            return None

        observed_at = payload.observed_at or datetime.now(timezone.utc)
        if alert == "normal":
            if self._enqueue_report_jobs:
                car_uuid = self._validate_car_id(car_id)
                async with self._sessions.begin() as session:
                    await self._advance_report_state(
                        session,
                        car_uuid=car_uuid,
                        risk_level=ALERT_RISK_NUMBER[alert],
                        observed_at=observed_at,
                    )
            return None

        car_uuid = self._validate_car_id(car_id)

        ml_risk_level = (
            ALERT_RISK_NUMBER[inference.ml_pattern_stage]
            if inference.ml_pattern_stage is not None
            else None
        )
        raw_metrics_payload = {
            "request": payload.model_dump(mode="json", exclude_none=True),
            "inference": inference.model_dump(mode="json", exclude_none=True),
        }
        if twin_frame is not None:
            raw_metrics_payload["twin_frame"] = twin_frame.model_dump(
                mode="json", exclude_none=True
            )
        raw_metrics = json.dumps(raw_metrics_payload, ensure_ascii=False)
        model_input = json.dumps(
            {
                "voltage_v": payload.voltage_v,
                "temp_mean_c": payload.temp_mean_c,
                "temp_max_c": payload.temp_max_c,
                "temp_delta_c": payload.temp_delta_c,
                "temp_saturation_fraction": payload.temp_saturation_fraction,
                "temp_saturation_all": payload.temp_saturation_all,
            },
            ensure_ascii=False,
        )

        async with self._sessions.begin() as session:
            report_transition = None
            if self._enqueue_report_jobs:
                report_transition = await self._advance_report_state(
                    session,
                    car_uuid=car_uuid,
                    risk_level=ALERT_RISK_NUMBER[alert],
                    observed_at=observed_at,
                )
            anomaly_id = await session.scalar(
                text(
                    '''
                    INSERT INTO public."ANOMALY_LOGS"
                        (abnormal_type, source_type, trigger_value, detected_at,
                         risk_level, car_id, session_id)
                    VALUES
                        (:abnormal_type, :source_type, :trigger_value, :detected_at,
                         :risk_level, CAST(:car_id AS uuid), CAST(:session_id AS uuid))
                    RETURNING anomaly_id
                    '''
                ),
                {
                    "abnormal_type": "BMS_SAFETY_ALERT",
                    "source_type": "FASTAPI_BMS",
                    "trigger_value": f"final_safety_alert={alert}",
                    "detected_at": observed_at,
                    "risk_level": ALERT_RISK_TEXT[alert],
                    "car_id": car_id,
                    "session_id": str(payload.session_id) if payload.session_id else None,
                },
            )
            await session.execute(
                text(
                    '''
                    INSERT INTO public."TWIN_FRAMES"
                        (observed_at, hotspot_cell_index, hotspot_connector_index,
                         ml_risk_level, physics_risk_level, final_risk_level,
                         image_risk_level, image_confidence, raw_metrics, model_input,
                         anomaly_id, car_id, session_id, source_image_ref)
                    VALUES
                        (:observed_at, :hotspot_cell_index, :hotspot_connector_index,
                         :ml_risk_level, :physics_risk_level, :final_risk_level,
                         :image_risk_level, :image_confidence, CAST(:raw_metrics AS jsonb),
                         CAST(:model_input AS jsonb), :anomaly_id, CAST(:car_id AS uuid),
                         CAST(:session_id AS uuid), :source_image_ref)
                    '''
                ),
                {
                    "observed_at": observed_at,
                    "hotspot_cell_index": payload.hotspot_cell_index,
                    "hotspot_connector_index": payload.hotspot_connector_index,
                    "ml_risk_level": ml_risk_level,
                    "physics_risk_level": ALERT_RISK_NUMBER[inference.physical_rule_level],
                    "final_risk_level": ALERT_RISK_NUMBER[alert],
                    "image_risk_level": payload.image_risk_level,
                    "image_confidence": payload.image_confidence,
                    "raw_metrics": raw_metrics,
                    "model_input": model_input,
                    "anomaly_id": anomaly_id,
                    "car_id": car_id,
                    "session_id": str(payload.session_id) if payload.session_id else None,
                    "source_image_ref": payload.source_image_ref,
                },
            )
            if report_transition is not None and report_transition.enqueue_report:
                incident_id = report_transition.state.incident_id
                if incident_id is None:
                    raise RuntimeError("report incident id is missing after confirmation")
                await session.execute(
                    text(
                        """
                        INSERT INTO ai_report_jobs (
                            job_id, job_key, job_type, car_id, anomaly_id,
                            status, available_at
                        ) VALUES (
                            :job_id, :job_key, 'ANOMALY', CAST(:car_id AS uuid),
                            :anomaly_id, 'PENDING', NOW()
                        )
                        ON CONFLICT (job_key) DO NOTHING
                        """
                    ),
                    {
                        "job_id": uuid4(),
                        "job_key": f"ANOMALY:{incident_id}:{alert}",
                        "car_id": car_id,
                        "anomaly_id": anomaly_id,
                    },
                )
        return str(anomaly_id)

    @staticmethod
    def _validate_car_id(car_id: str) -> UUID:
        try:
            return UUID(car_id)
        except ValueError as exc:
            raise ValueError(
                "vehicle_id must be the CAR.car_id UUID when persistence is enabled"
            ) from exc

    @staticmethod
    async def _advance_report_state(
        session: AsyncSession,
        *,
        car_uuid: UUID,
        risk_level: int,
        observed_at: datetime,
    ) -> ReportAlertTransition:
        # The row lock serializes samples for one vehicle across all FastAPI pods.
        await session.execute(
            text(
                """
                INSERT INTO ai_report_alert_states (car_id, last_observed_at)
                VALUES (:car_id, :last_observed_at)
                ON CONFLICT (car_id) DO NOTHING
                """
            ),
            {"car_id": car_uuid, "last_observed_at": observed_at},
        )
        result = await session.execute(
            text(
                """
                SELECT incident_id, current_risk_level, last_reported_risk_level,
                       candidate_risk_level, candidate_count, normal_since,
                       last_observed_at
                FROM ai_report_alert_states
                WHERE car_id = :car_id
                FOR UPDATE
                """
            ),
            {"car_id": car_uuid},
        )
        row = result.mappings().one()
        transition = advance_report_alert_state(
            ReportAlertState(
                incident_id=row["incident_id"],
                current_risk_level=row["current_risk_level"],
                last_reported_risk_level=row["last_reported_risk_level"],
                candidate_risk_level=row["candidate_risk_level"],
                candidate_count=row["candidate_count"],
                normal_since=row["normal_since"],
                last_observed_at=row["last_observed_at"],
            ),
            risk_level=risk_level,
            observed_at=observed_at,
        )
        next_state = transition.state
        await session.execute(
            text(
                """
                UPDATE ai_report_alert_states
                SET incident_id = :incident_id,
                    current_risk_level = :current_risk_level,
                    last_reported_risk_level = :last_reported_risk_level,
                    candidate_risk_level = :candidate_risk_level,
                    candidate_count = :candidate_count,
                    normal_since = :normal_since,
                    last_observed_at = :last_observed_at,
                    updated_at = NOW()
                WHERE car_id = :car_id
                """
            ),
            {
                "car_id": car_uuid,
                "incident_id": next_state.incident_id,
                "current_risk_level": next_state.current_risk_level,
                "last_reported_risk_level": next_state.last_reported_risk_level,
                "candidate_risk_level": next_state.candidate_risk_level,
                "candidate_count": next_state.candidate_count,
                "normal_since": next_state.normal_since,
                "last_observed_at": next_state.last_observed_at,
            },
        )
        return transition
