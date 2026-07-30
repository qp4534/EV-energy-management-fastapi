from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable


class TimestampConflict(ValueError):
    pass


@dataclass
class VehicleSession:
    supervisor: Any
    lock: asyncio.Lock
    last_timestamp: float | None = None
    last_access_monotonic: float = 0.0


class SessionManager:
    def __init__(self, factory: Callable[[], Any], ttl_seconds: int, max_sessions: int) -> None:
        self._factory, self._ttl, self._max = factory, ttl_seconds, max_sessions
        self._sessions: dict[str, VehicleSession] = {}
        self._guard = asyncio.Lock()

    async def _get(self, vehicle_id: str) -> VehicleSession:
        async with self._guard:
            now = time.monotonic()
            for key in [k for k, v in self._sessions.items() if now - v.last_access_monotonic > self._ttl]:
                del self._sessions[key]
            session = self._sessions.get(vehicle_id)
            if session is None:
                if len(self._sessions) >= self._max:
                    oldest = min(self._sessions, key=lambda k: self._sessions[k].last_access_monotonic)
                    del self._sessions[oldest]
                session = VehicleSession(self._factory(), asyncio.Lock(), last_access_monotonic=now)
                self._sessions[vehicle_id] = session
            return session

    async def push(self, vehicle_id: str, sample: dict[str, Any], timestamp: float) -> Any:
        session = await self._get(vehicle_id)
        async with session.lock:
            if session.last_timestamp is not None and timestamp <= session.last_timestamp:
                raise TimestampConflict("timestamp_seconds must be strictly increasing for a vehicle session")
            if session.last_timestamp is not None and abs(timestamp - session.last_timestamp - 1.0) > 1e-6:
                session.supervisor.reset()
            result = session.supervisor.push(sample, timestamp)
            session.last_timestamp, session.last_access_monotonic = timestamp, time.monotonic()
            return result

    async def reset(self, vehicle_id: str) -> bool:
        session = await self._get(vehicle_id)
        async with session.lock:
            session.supervisor.reset(); session.last_timestamp = None; session.last_access_monotonic = time.monotonic()
        return True

    async def delete(self, vehicle_id: str) -> bool:
        async with self._guard:
            return self._sessions.pop(vehicle_id, None) is not None
