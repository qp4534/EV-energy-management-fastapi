from __future__ import annotations

from typing import Any

import httpx


class VehicleStateUnavailable(RuntimeError):
    pass


class InferenceVehicleStateClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 2.0,
        internal_token: str = "",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.internal_token = internal_token
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def latest(self, vehicle_id: str) -> dict[str, Any] | None:
        headers = {}
        if self.internal_token:
            headers["X-Internal-Token"] = self.internal_token
        try:
            response = await self._client.get(
                f"{self.base_url}/api/v1/twins/vehicles/{vehicle_id}/latest",
                headers=headers,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise VehicleStateUnavailable("invalid latest vehicle response")
            return payload
        except VehicleStateUnavailable:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise VehicleStateUnavailable("vehicle state service unavailable") from exc
