from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.twin_redis import TwinRedisStore
from app.db.models import (
    INCIDENT_COMPLETE,
    INCIDENT_INCOMPLETE,
    INCIDENT_OPEN,
    INCIDENT_TYPE_BATTERY,
    INCIDENT_TYPE_CONNECTOR,
    INCIDENT_TYPE_GENERAL,
    TwinIncident,
)
from app.db.repository import TwinRepository
from app.db.session import create_database
from app.schemas.twins import TwinFrame


LOGGER = logging.getLogger("twin-persistence")
HEARTBEAT_PATH = Path("/tmp/twin-worker-heartbeat")
PRE_SECONDS = 3_600
POST_SECONDS = 7_200
REARM_SECONDS = 60
INCIDENT_NAMESPACE = uuid.UUID("08b1619f-b710-4ab4-b104-f51d1b9b7e57")


def deterministic_incident_id(vehicle_id: str, triggered_at: datetime) -> str:
    utc = triggered_at.astimezone(timezone.utc).isoformat(timespec="microseconds")
    return str(uuid.uuid5(INCIDENT_NAMESPACE, f"{vehicle_id}|{utc}"))


def frames_are_contiguous(frames: list[TwinFrame]) -> bool:
    return all(
        right.sequence == left.sequence + 1
        and abs((right.observed_at - left.observed_at).total_seconds() - 1.0) < 1e-6
        for left, right in zip(frames, frames[1:])
    )


def incident_type_for_frame(frame: TwinFrame) -> str:
    connector_level = max(frame.connector_state_level, default=0)
    battery_level = max(frame.state_level, default=0)
    if connector_level <= 0 and battery_level <= 0:
        return INCIDENT_TYPE_GENERAL
    if connector_level >= battery_level:
        return INCIDENT_TYPE_CONNECTOR
    return INCIDENT_TYPE_BATTERY


class IncidentProcessor:
    def __init__(self, redis_store: TwinRedisStore, repository: TwinRepository) -> None:
        self.redis = redis_store
        self.repository = repository

    async def process(self, session: AsyncSession, frame: TwinFrame) -> None:
        active = await self.repository.active_incidents(session, frame.vehicle_id)
        for incident in active:
            if frame.observed_at >= incident.window_end:
                await self._finish(session, incident)
                continue
            if frame.observed_at >= incident.triggered_at:
                await self.repository.insert_frames(session, incident.id, [frame])
            await self._maybe_rearm(session, incident, frame)
            if frame.observed_at + timedelta(seconds=1) >= incident.window_end:
                await self._finish(session, incident)

        latest = await self.repository.latest_incident(session, frame.vehicle_id)
        if latest is not None and latest.rearmed_at is None:
            await self._maybe_rearm(session, latest, frame)
            if latest.rearmed_at is None:
                return

        if frame.final_risk_level < 1:
            return
        if (
            latest is not None
            and latest.rearmed_at is not None
            and frame.observed_at <= latest.rearmed_at
        ):
            # A reclaimed/pending message from before the durable re-arm point
            # may still have an adjacent historical risk frame in prebuffer.
            # It must not manufacture a new incident after newer state has
            # already been committed.
            return
        previous = await self.redis.frames_before(
            frame.vehicle_id,
            frame.observed_at,
            limit=PRE_SECONDS,
        )
        if not previous:
            return
        prior = previous[-1]
        if (
            prior.final_risk_level < 1
            or prior.sequence + 1 != frame.sequence
            or abs((frame.observed_at - prior.observed_at).total_seconds() - 1.0)
            >= 1e-6
        ):
            return

        triggered_at = frame.observed_at
        window_start = triggered_at - timedelta(seconds=PRE_SECONDS)
        window_end = triggered_at + timedelta(seconds=POST_SECONDS)
        pre_frames = [
            item
            for item in previous
            if window_start <= item.observed_at < triggered_at
        ][-PRE_SECONDS:]
        pre_frames = self._contiguous_suffix(pre_frames, frame)
        incident_id = deterministic_incident_id(frame.vehicle_id, triggered_at)
        await self.repository.create_incident(
            session,
            incident_id=incident_id,
            vehicle_id=frame.vehicle_id,
            triggered_at=triggered_at,
            window_start=window_start,
            window_end=window_end,
            incident_type=incident_type_for_frame(frame),
        )
        await self.repository.insert_frames(
            session,
            incident_id,
            [*pre_frames, frame],
        )

    async def _finish(
        self, session: AsyncSession, incident: TwinIncident
    ) -> None:
        count = await self.repository.count_frames(session, incident.id)
        status = (
            INCIDENT_COMPLETE
            if count == PRE_SECONDS + POST_SECONDS
            else INCIDENT_INCOMPLETE
        )
        await self.repository.mark_finished(
            session,
            incident.id,
            incident.window_end,
            status,
        )
        incident.status = status

    @staticmethod
    def _contiguous_suffix(
        previous: list[TwinFrame], current: TwinFrame
    ) -> list[TwinFrame]:
        suffix: list[TwinFrame] = []
        right = current
        for candidate in reversed(previous):
            if (
                candidate.sequence + 1 != right.sequence
                or abs(
                    (right.observed_at - candidate.observed_at).total_seconds()
                    - 1.0
                )
                >= 1e-6
            ):
                break
            suffix.append(candidate)
            right = candidate
        suffix.reverse()
        return suffix

    async def _maybe_rearm(
        self,
        session: AsyncSession,
        incident: TwinIncident,
        frame: TwinFrame,
    ) -> None:
        if incident.rearmed_at is not None or frame.final_risk_level != 0:
            return
        previous = await self.redis.frames_before(
            frame.vehicle_id,
            frame.observed_at,
            limit=REARM_SECONDS - 1,
        )
        run = [*previous[-(REARM_SECONDS - 1) :], frame]
        if (
            len(run) == REARM_SECONDS
            and run[0].observed_at >= incident.triggered_at
            and all(item.final_risk_level == 0 for item in run)
            and frames_are_contiguous(run)
        ):
            await self.repository.mark_rearmed(
                session, incident.id, frame.observed_at
            )
            incident.rearmed_at = frame.observed_at


def write_heartbeat() -> None:
    HEARTBEAT_PATH.write_text(str(time.time()), encoding="ascii")


async def run_worker() -> None:
    settings = Settings.load()
    engine, sessions = create_database(settings.database_url)
    redis = Redis.from_url(settings.redis_url, decode_responses=False)
    store = TwinRedisStore(redis)
    repository = TwinRepository()
    processor = IncidentProcessor(store, repository)
    await store.ping()
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    await store.ensure_consumer_group(settings.twin_consumer_group)
    last_claim = 0.0
    write_heartbeat()
    try:
        while True:
            messages: list[tuple[str, TwinFrame]] = []
            now = time.monotonic()
            if now - last_claim >= 30.0:
                messages.extend(
                    await store.claim_stale(
                        settings.twin_consumer_group,
                        settings.twin_consumer_name,
                    )
                )
                last_claim = now
            messages.extend(
                await store.read_group(
                    settings.twin_consumer_group,
                    settings.twin_consumer_name,
                )
            )
            batch_failed = False
            for message_id, frame in messages:
                try:
                    async with sessions() as session:
                        async with session.begin():
                            await processor.process(session, frame)
                    await store.acknowledge(
                        settings.twin_consumer_group, message_id
                    )
                    write_heartbeat()
                except Exception:
                    batch_failed = True
                    LOGGER.exception("failed to persist TwinFrame %s", message_id)
            if batch_failed:
                continue
            try:
                await store.ping()
                async with engine.connect() as connection:
                    await connection.execute(text("SELECT 1"))
                write_heartbeat()
            except Exception:
                LOGGER.exception("worker dependency health check failed")
    finally:
        await redis.aclose()
        await engine.dispose()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
