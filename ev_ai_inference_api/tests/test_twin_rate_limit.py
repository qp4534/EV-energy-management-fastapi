import pytest
from fastapi import HTTPException

from app.core.twin_rate_limit import TwinRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_is_scoped_and_returns_retry_after() -> None:
    limiter = TwinRateLimiter()
    await limiter.check("read", "user-a", limit=2, window_seconds=60)
    await limiter.check("read", "user-a", limit=2, window_seconds=60)
    await limiter.check("read", "user-b", limit=2, window_seconds=60)

    with pytest.raises(HTTPException) as raised:
        await limiter.check("read", "user-a", limit=2, window_seconds=60)

    assert raised.value.status_code == 429
    assert int(raised.value.headers["Retry-After"]) > 0
