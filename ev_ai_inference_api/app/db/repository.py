from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import math

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.twins import IncidentSummary, TwinFrame

from .models import (
    INCIDENT_COMPLETE,
    INCIDENT_INCOMPLETE,
    INCIDENT_OPEN,
    INCIDENT_TYPE_GENERAL,
    INCIDENT_TYPES,
    TwinFrameRecord,
    TwinIncident,
)


def twin_frame_values(incident_id: str, frame: TwinFrame) -> dict[str, object]:
    return {
        "incident_id": incident_id,
        "observed_at": frame.observed_at,
        "sequence": frame.sequence,
        "schema_version": frame.schema_version,
        "layout_id": frame.layout_id,
        "temperature_decic": frame.temperature_decic,
        "voltage_mv": frame.voltage_mv,
        "state_level": frame.state_level,
        "connector_temperature_decic": frame.connector_temperature_decic,
        "connector_state_level": frame.connector_state_level,
        "hotspot_cell_index": frame.hotspot_cell_index,
        "hotspot_connector_index": frame.hotspot_connector_index,
        "ml_risk_level": frame.ml_risk_level,
        "physics_risk_level": frame.physics_risk_level,
        "final_risk_level": frame.final_risk_level,
        "cell_heat_score": frame.cell_heat_score,
        "image_risk_level": frame.image_risk_level,
        "image_confidence": frame.image_confidence,
        "image_probabilities": frame.image_probabilities,
        "image_model_status": frame.image_model_status,
        "module_heat_score": frame.module_heat_score,
        "module_state_level": frame.module_state_level,
        "hotspot_module_index": frame.hotspot_module_index,
        "thermal_frame_ref": frame.thermal_frame_ref,
        "thermal_frame_sha256": frame.thermal_frame_sha256,
        "fusion_source": frame.fusion_source,
    }


def record_to_frame(record: TwinFrameRecord, vehicle_id: str) -> TwinFrame:
    return TwinFrame(
        schema_version=record.schema_version,
        layout_id=record.layout_id,
        vehicle_id=vehicle_id,
        observed_at=record.observed_at,
        sequence=record.sequence,
        temperature_decic=list(record.temperature_decic),
        voltage_mv=list(record.voltage_mv),
        state_level=list(record.state_level),
        connector_temperature_decic=list(record.connector_temperature_decic),
        connector_state_level=list(record.connector_state_level),
        hotspot_cell_index=record.hotspot_cell_index,
        hotspot_connector_index=record.hotspot_connector_index,
        ml_risk_level=record.ml_risk_level,
        physics_risk_level=record.physics_risk_level,
        final_risk_level=record.final_risk_level,
        cell_heat_score=(
            list(record.cell_heat_score)
            if record.cell_heat_score is not None
            else None
        ),
        image_risk_level=record.image_risk_level,
        image_confidence=record.image_confidence,
        image_probabilities=(
            list(record.image_probabilities)
            if record.image_probabilities is not None
            else None
        ),
        image_model_status=record.image_model_status or "unavailable",
        module_heat_score=(
            list(record.module_heat_score)
            if record.module_heat_score is not None
            else None
        ),
        module_state_level=(
            list(record.module_state_level)
            if record.module_state_level is not None
            else None
        ),
        hotspot_module_index=record.hotspot_module_index,
        thermal_frame_ref=record.thermal_frame_ref,
        thermal_frame_sha256=record.thermal_frame_sha256,
        fusion_source=record.fusion_source or "sensor-only",
    )


class TwinRepository:
    async def latest_abnormal_type_for_car(
        self,
        session: AsyncSession,
        car_id: str,
    ) -> str | None:
        return await session.scalar(
            text(
                """
                SELECT a."abnormal_type"
                FROM "ANOMALY_LOGS" AS a
                WHERE a."car_id" = :car_id
                ORDER BY a."detected_at" DESC
                LIMIT 1
                """
            ),
            {"car_id": car_id},
        )

    async def create_incident(
        self,
        session: AsyncSession,
        *,
        incident_id: str,
        vehicle_id: str,
        triggered_at: datetime,
        window_start: datetime,
        window_end: datetime,
        incident_type: str = INCIDENT_TYPE_GENERAL,
    ) -> None:
        if incident_type not in INCIDENT_TYPES:
            raise ValueError(f"unsupported incident_type: {incident_type}")
        statement = (
            pg_insert(TwinIncident)
            .values(
                id=incident_id,
                vehicle_id=vehicle_id,
                triggered_at=triggered_at,
                window_start=window_start,
                window_end=window_end,
                incident_type=incident_type,
                status=INCIDENT_OPEN,
            )
            .on_conflict_do_nothing()
        )
        await session.execute(statement)

    async def insert_frames(
        self,
        session: AsyncSession,
        incident_id: str,
        frames: Sequence[TwinFrame],
        *,
        chunk_size: int = 500,
    ) -> None:
        for offset in range(0, len(frames), chunk_size):
            values = [
                twin_frame_values(incident_id, frame)
                for frame in frames[offset : offset + chunk_size]
            ]
            if values:
                await session.execute(
                    pg_insert(TwinFrameRecord)
                    .values(values)
                    .on_conflict_do_nothing()
                )

    async def replace_seed_frames(
        self,
        session: AsyncSession,
        incident_id: str,
        frames: Sequence[TwinFrame],
    ) -> None:
        """Atomically refresh only deterministic local-simulator incident frames."""

        await session.execute(
            delete(TwinFrameRecord).where(
                TwinFrameRecord.incident_id == incident_id
            )
        )
        await self.insert_frames(session, incident_id, frames)

    async def latest_incident(
        self, session: AsyncSession, vehicle_id: str
    ) -> TwinIncident | None:
        return await session.scalar(
            select(TwinIncident)
            .where(TwinIncident.vehicle_id == vehicle_id)
            .order_by(TwinIncident.triggered_at.desc())
            .limit(1)
        )

    async def latest_complete_incident(
        self, session: AsyncSession, vehicle_id: str
    ) -> TwinIncident | None:
        return await session.scalar(
            select(TwinIncident)
            .where(
                TwinIncident.vehicle_id == vehicle_id,
                TwinIncident.status == INCIDENT_COMPLETE,
            )
            .order_by(TwinIncident.triggered_at.desc())
            .limit(1)
        )

    async def complete_incident(
        self,
        session: AsyncSession,
        vehicle_id: str,
        incident_id: str,
    ) -> TwinIncident | None:
        return await session.scalar(
            select(TwinIncident).where(
                TwinIncident.id == incident_id,
                TwinIncident.vehicle_id == vehicle_id,
                TwinIncident.status == INCIDENT_COMPLETE,
            )
        )

    async def active_incidents(
        self, session: AsyncSession, vehicle_id: str
    ) -> list[TwinIncident]:
        return list(
            (
                await session.scalars(
                    select(TwinIncident)
                    .where(
                        TwinIncident.vehicle_id == vehicle_id,
                        TwinIncident.status == INCIDENT_OPEN,
                    )
                    .order_by(TwinIncident.triggered_at)
                )
            ).all()
        )

    async def get_incident(
        self, session: AsyncSession, incident_id: str
    ) -> TwinIncident | None:
        return await session.get(TwinIncident, incident_id)

    async def mark_finished(
        self,
        session: AsyncSession,
        incident_id: str,
        completed_at: datetime,
        status: int,
    ) -> None:
        if status not in {INCIDENT_COMPLETE, INCIDENT_INCOMPLETE}:
            raise ValueError("finished incident status must be complete or incomplete")
        await session.execute(
            update(TwinIncident)
            .where(TwinIncident.id == incident_id)
            .values(status=status, completed_at=completed_at)
        )

    async def mark_complete(
        self, session: AsyncSession, incident_id: str, completed_at: datetime
    ) -> None:
        await self.mark_finished(
            session, incident_id, completed_at, INCIDENT_COMPLETE
        )

    async def mark_rearmed(
        self, session: AsyncSession, incident_id: str, rearmed_at: datetime
    ) -> None:
        await session.execute(
            update(TwinIncident)
            .where(
                TwinIncident.id == incident_id,
                TwinIncident.rearmed_at.is_(None),
            )
            .values(rearmed_at=rearmed_at)
        )

    async def count_frames(self, session: AsyncSession, incident_id: str) -> int:
        return int(
            await session.scalar(
                select(func.count(TwinFrameRecord.id)).where(
                    TwinFrameRecord.incident_id == incident_id
                )
            )
            or 0
        )

    async def list_incidents(
        self, session: AsyncSession, vehicle_id: str
    ) -> list[IncidentSummary]:
        counts = (
            select(
                TwinFrameRecord.incident_id.label("incident_id"),
                func.count(TwinFrameRecord.id).label("frame_count"),
            )
            .group_by(TwinFrameRecord.incident_id)
            .subquery()
        )
        rows = (
            await session.execute(
                select(TwinIncident, func.coalesce(counts.c.frame_count, 0))
                .outerjoin(counts, counts.c.incident_id == TwinIncident.id)
                .where(TwinIncident.vehicle_id == vehicle_id)
                .order_by(TwinIncident.triggered_at.desc())
            )
        ).all()
        return [self._summary(incident, int(count)) for incident, count in rows]

    async def latest_incident_summary(
        self, session: AsyncSession, vehicle_id: str
    ) -> IncidentSummary | None:
        incident = await self.latest_incident(session, vehicle_id)
        if incident is None:
            return None
        return self._summary(
            incident,
            await self.count_frames(session, incident.id),
        )

    async def history(
        self,
        session: AsyncSession,
        incident: TwinIncident,
        resolution_seconds: int,
    ) -> list[TwinFrame]:
        bucket = func.floor(
            func.extract(
                "epoch", TwinFrameRecord.observed_at - incident.window_start
            )
            / resolution_seconds
        )
        bucket_count = math.ceil(
            (incident.window_end - incident.window_start).total_seconds()
            / resolution_seconds
        )
        ranked = (
            select(
                TwinFrameRecord.id.label("frame_id"),
                func.row_number()
                .over(
                    partition_by=bucket,
                    order_by=(
                        TwinFrameRecord.observed_at.desc(),
                        TwinFrameRecord.sequence.desc(),
                    ),
                )
                .label("row_number"),
            )
            .where(
                TwinFrameRecord.incident_id == incident.id,
                TwinFrameRecord.observed_at >= incident.window_start,
                TwinFrameRecord.observed_at < incident.window_end,
                bucket >= 0,
                bucket < bucket_count,
            )
            .subquery()
        )
        records = (
            await session.scalars(
                select(TwinFrameRecord)
                .join(ranked, ranked.c.frame_id == TwinFrameRecord.id)
                .where(ranked.c.row_number == 1)
                .order_by(TwinFrameRecord.observed_at)
            )
        ).all()
        return [record_to_frame(record, incident.vehicle_id) for record in records]

    @staticmethod
    def _summary(incident: TwinIncident, frame_count: int) -> IncidentSummary:
        return IncidentSummary(
            id=incident.id,
            vehicle_id=incident.vehicle_id,
            incident_type=incident.incident_type,
            triggered_at=incident.triggered_at,
            window_start=incident.window_start,
            window_end=incident.window_end,
            status=(
                "complete"
                if incident.status == INCIDENT_COMPLETE
                else "incomplete"
                if incident.status == INCIDENT_INCOMPLETE
                else "open"
            ),
            frame_count=frame_count,
            rearmed_at=incident.rearmed_at,
        )
