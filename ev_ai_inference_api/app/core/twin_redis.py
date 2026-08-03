from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from redis.asyncio import Redis
from redis.asyncio.client import PubSub

from app.schemas.twins import RiskVehicleItem, TwinFrame


LATEST_TTL_SECONDS = 300
PREBUFFER_MAX_FRAMES = 3_601
PERSIST_STREAM = "twin:persist"
RISK_SORTED_SET = "twin:risk"


def latest_key(vehicle_id: str) -> str:
    return f"twin:latest:{vehicle_id}"


def prebuffer_key(vehicle_id: str) -> str:
    return f"twin:prebuffer:{vehicle_id}"


def live_channel(vehicle_id: str) -> str:
    return f"twin:live:{vehicle_id}"


def _text(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


class TwinRedisStore:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def ping(self) -> bool:
        return bool(await self.redis.ping())

    async def publish(self, frame: TwinFrame) -> None:
        payload = frame.model_dump_json()
        pipe = self.redis.pipeline(transaction=True)
        pipe.set(latest_key(frame.vehicle_id), payload, ex=LATEST_TTL_SECONDS)
        pipe.xadd(
            prebuffer_key(frame.vehicle_id),
            {"frame": payload},
            maxlen=PREBUFFER_MAX_FRAMES,
            approximate=False,
        )
        pipe.xadd(
            PERSIST_STREAM,
            {
                "vehicle_id": frame.vehicle_id,
                "observed_at": frame.observed_at.isoformat(),
                "sequence": str(frame.sequence),
                "frame": payload,
            },
        )
        pipe.publish(live_channel(frame.vehicle_id), payload)
        if frame.final_risk_level >= 1:
            pipe.zadd(RISK_SORTED_SET, {frame.vehicle_id: frame.final_risk_level})
        else:
            pipe.zrem(RISK_SORTED_SET, frame.vehicle_id)
        await pipe.execute()

    async def seed_latest(self, frame: TwinFrame) -> None:
        """Populate current/risk views without enqueueing already-persisted seed data."""

        payload = frame.model_dump_json()
        pipe = self.redis.pipeline(transaction=True)
        pipe.set(latest_key(frame.vehicle_id), payload, ex=LATEST_TTL_SECONDS)
        if frame.final_risk_level >= 1:
            pipe.zadd(RISK_SORTED_SET, {frame.vehicle_id: frame.final_risk_level})
        else:
            pipe.zrem(RISK_SORTED_SET, frame.vehicle_id)
        pipe.publish(live_channel(frame.vehicle_id), payload)
        await pipe.execute()

    async def get_latest(self, vehicle_id: str) -> TwinFrame | None:
        payload = await self.redis.get(latest_key(vehicle_id))
        if payload is None:
            return None
        return TwinFrame.model_validate_json(payload)

    async def risk_vehicles(self) -> list[RiskVehicleItem]:
        members = await self.redis.zrevrange(RISK_SORTED_SET, 0, -1)
        if not members:
            return []
        vehicle_ids = [_text(member) for member in members]
        payloads = await self.redis.mget([latest_key(item) for item in vehicle_ids])
        stale: list[str] = []
        items: list[RiskVehicleItem] = []
        for vehicle_id, payload in zip(vehicle_ids, payloads, strict=True):
            if payload is None:
                stale.append(vehicle_id)
                continue
            frame = TwinFrame.model_validate_json(payload)
            if frame.final_risk_level < 1:
                stale.append(vehicle_id)
                continue
            items.append(
                RiskVehicleItem(
                    vehicle_id=vehicle_id,
                    observed_at=frame.observed_at,
                    sequence=frame.sequence,
                    final_risk_level=frame.final_risk_level,
                )
            )
        if stale:
            await self.redis.zrem(RISK_SORTED_SET, *stale)
        return sorted(
            items,
            key=lambda item: (
                -item.final_risk_level,
                -item.observed_at.timestamp(),
                item.vehicle_id,
            ),
        )

    async def frames_before(
        self,
        vehicle_id: str,
        observed_at,
        *,
        limit: int,
    ) -> list[TwinFrame]:
        entries = await self.redis.xrevrange(
            prebuffer_key(vehicle_id),
            max="+",
            min="-",
            count=PREBUFFER_MAX_FRAMES,
        )
        frames: list[TwinFrame] = []
        for _, fields in entries:
            payload = fields.get(b"frame")
            if payload is None:
                payload = fields.get("frame")
            if payload is None:
                continue
            frame = TwinFrame.model_validate_json(payload)
            if frame.observed_at < observed_at:
                frames.append(frame)
        frames.sort(key=lambda item: (item.observed_at, item.sequence))
        return frames[-limit:]

    async def subscribe(self, vehicle_id: str) -> PubSub:
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(live_channel(vehicle_id))
        return pubsub

    async def live_messages(self, pubsub: PubSub) -> AsyncIterator[TwinFrame]:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            yield TwinFrame.model_validate_json(message["data"])

    async def close_subscription(self, pubsub: PubSub) -> None:
        await pubsub.unsubscribe()
        await pubsub.aclose()

    async def ensure_consumer_group(self, group: str) -> None:
        try:
            await self.redis.xgroup_create(
                PERSIST_STREAM,
                group,
                id="0-0",
                mkstream=True,
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def read_group(
        self,
        group: str,
        consumer: str,
        *,
        count: int = 100,
        block_ms: int = 1_000,
    ) -> list[tuple[str, TwinFrame]]:
        batches = await self.redis.xreadgroup(
            group,
            consumer,
            {PERSIST_STREAM: ">"},
            count=count,
            block=block_ms,
        )
        return self._decode_batches(batches)

    async def claim_stale(
        self,
        group: str,
        consumer: str,
        *,
        min_idle_ms: int = 60_000,
        count: int = 100,
    ) -> list[tuple[str, TwinFrame]]:
        result = await self.redis.xautoclaim(
            PERSIST_STREAM,
            group,
            consumer,
            min_idle_ms,
            "0-0",
            count=count,
        )
        entries = result[1] if len(result) >= 2 else []
        return self._decode_entries(entries)

    async def acknowledge(self, group: str, message_id: str) -> None:
        pipe = self.redis.pipeline(transaction=True)
        pipe.xack(PERSIST_STREAM, group, message_id)
        pipe.xdel(PERSIST_STREAM, message_id)
        await pipe.execute()

    @classmethod
    def _decode_batches(cls, batches: list[Any]) -> list[tuple[str, TwinFrame]]:
        decoded: list[tuple[str, TwinFrame]] = []
        for _, entries in batches:
            decoded.extend(cls._decode_entries(entries))
        return decoded

    @staticmethod
    def _decode_entries(entries: list[Any]) -> list[tuple[str, TwinFrame]]:
        decoded: list[tuple[str, TwinFrame]] = []
        for message_id, fields in entries:
            payload = fields.get(b"frame")
            if payload is None:
                payload = fields.get("frame")
            if payload is None:
                continue
            decoded.append(
                (_text(message_id), TwinFrame.model_validate_json(payload))
            )
        return decoded
