from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import time

import httpx

from app.schemas.twins import ThermalInferenceResult


@dataclass
class ThermalInferenceClient:
    base_url: str = ""
    token: str = ""
    timeout_seconds: float = 0.8
    retry_cooldown_seconds: float = 10.0
    _retry_after: float = field(default=0.0, init=False, repr=False)

    @property
    def enabled(self) -> bool:
        return bool(self.base_url.strip())

    async def infer(
        self,
        *,
        vehicle_id: str,
        observed_at: str,
        sequence: int,
        layout_id: str,
        image_bytes: bytes,
    ) -> ThermalInferenceResult:
        if not self.enabled or time.monotonic() < self._retry_after:
            return ThermalInferenceResult(status="unavailable")
        headers = {"X-Thermal-Worker-Token": self.token} if self.token else {}
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                timeout=self.timeout_seconds,
            ) as client:
                response = await client.post(
                    "/v1/thermal/infer",
                    data={
                        "vehicle_id": vehicle_id,
                        "observed_at": observed_at,
                        "sequence": str(sequence),
                        "layout_id": layout_id,
                    },
                    files={
                        "thermal_image": (
                            f"{vehicle_id}-{sequence}.png",
                            image_bytes,
                            "image/png",
                        )
                    },
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
            return ThermalInferenceResult.model_validate(payload)
        except (httpx.HTTPError, ValueError, TypeError):
            self._retry_after = time.monotonic() + self.retry_cooldown_seconds
            return ThermalInferenceResult(status="unavailable")
