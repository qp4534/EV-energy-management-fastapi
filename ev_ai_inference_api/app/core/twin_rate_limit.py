from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from fastapi import HTTPException


@dataclass
class _Bucket:
    started_at: float
    count: int


class TwinRateLimiter:
    """Bounded single-Pod limiter for the currently stateful Twin API."""

    def __init__(self, max_buckets: int = 10_000) -> None:
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._lock = asyncio.Lock()
        self._max_buckets = max_buckets

    async def check(
        self,
        scope: str,
        identity: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> None:
        now = time.monotonic()
        key = (scope, identity)
        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None or now - bucket.started_at >= window_seconds:
                self._buckets[key] = _Bucket(now, 1)
            else:
                bucket.count += 1
                if bucket.count > limit:
                    retry_after = max(
                        1, int(window_seconds - (now - bucket.started_at)) + 1
                    )
                    raise HTTPException(
                        status_code=429,
                        detail="Twin API rate limit exceeded",
                        headers={"Retry-After": str(retry_after)},
                    )
            if len(self._buckets) > self._max_buckets:
                oldest = min(
                    self._buckets,
                    key=lambda item: self._buckets[item].started_at,
                )
                if oldest != key:
                    self._buckets.pop(oldest, None)
