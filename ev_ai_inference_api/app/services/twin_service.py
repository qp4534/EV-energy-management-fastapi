from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.twin_redis import TwinRedisStore
from app.db.anomaly_persistence import AnomalyPersistence
from app.db.repository import TwinRepository
from app.schemas.current_stage import SampleRequest
from app.schemas.twins import (
    IncidentListResponse,
    RiskVehicleListResponse,
    TwinFrame,
    TwinHistoryResponse,
    TwinSampleRequest,
)

from .current_stage_service import CurrentStageService
from .thermal_inference import ThermalInferenceClient
from .thermal_render import analyze_cell_heat_scores
from .twin_fusion import fuse_twin_state


_VEHICLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_STAGE_LEVEL = {
    "normal": 0,
    "caution": 1,
    "warning": 2,
    "emergency": 3,
    "unknown": 0,
}
_LEVEL_STAGE = {
    0: "normal",
    1: "caution",
    2: "warning",
    3: "emergency",
}


class InvalidVehicleId(ValueError):
    pass


class TwinSequenceConflict(ValueError):
    pass


class IncidentNotFound(LookupError):
    pass


def validate_vehicle_id(vehicle_id: str) -> str:
    if not _VEHICLE_ID.fullmatch(vehicle_id):
        raise InvalidVehicleId(
            "vehicle_id must be 1-128 URL-safe letters, digits, dots, colons, underscores, or hyphens"
        )
    return vehicle_id


def temperature_level(temperature_decic: int) -> int:
    if temperature_decic >= 800:
        return 3
    if temperature_decic >= 600:
        return 2
    if temperature_decic >= 450:
        return 1
    return 0


def voltage_level(voltage_mv: int) -> int:
    if voltage_mv <= 1_500 or voltage_mv >= 4_500:
        return 3
    if voltage_mv <= 2_000 or voltage_mv >= 4_400:
        return 2
    if voltage_mv < 2_400 or voltage_mv > 4_350:
        return 1
    return 0


def _model_request(payload: TwinSampleRequest) -> SampleRequest:
    temperatures_c = [value / 10.0 for value in payload.temperature_decic]
    voltages_v = [value / 1_000.0 for value in payload.voltage_mv]
    maximum = max(temperatures_c)
    minimum = min(temperatures_c)
    saturated = sum(value >= 150.0 for value in temperatures_c)
    return SampleRequest(
        timestamp_seconds=payload.observed_at.timestamp(),
        session_id=payload.session_id,
        voltage_v=sum(voltages_v) / len(voltages_v),
        temp_mean_c=sum(temperatures_c) / len(temperatures_c),
        temp_max_c=maximum,
        temp_delta_c=maximum - minimum,
        temp_saturation_fraction=saturated / len(temperatures_c),
        temp_saturation_all=saturated == len(temperatures_c),
        raw_temp_max_c=maximum,
        raw_temp_mean_c=sum(temperatures_c) / len(temperatures_c),
        ambient_temp_c=payload.ambient_temperature_c,
        pack_current_a=payload.pack_current_a,
        cell_voltages_v=voltages_v,
        temperature_decic=list(payload.temperature_decic),
        connector_temperature_decic=list(payload.connector_temperature_decic),
        charging_gun_temperature_c=max(payload.connector_temperature_decic) / 10.0,
        observed_at=payload.observed_at,
    )


class TwinService:
    def __init__(
        self,
        current_stage: CurrentStageService,
        redis_store: TwinRedisStore,
        sessions: async_sessionmaker[AsyncSession],
        repository: TwinRepository | None = None,
        thermal_inference: ThermalInferenceClient | None = None,
        minimum_image_confidence: float = 0.70,
        anomaly_persistence: AnomalyPersistence | None = None,
    ) -> None:
        self.current_stage = current_stage
        self.redis = redis_store
        self.sessions = sessions
        self.repository = repository or TwinRepository()
        self.thermal_inference = thermal_inference or ThermalInferenceClient()
        self.minimum_image_confidence = minimum_image_confidence
        self.anomaly_persistence = anomaly_persistence
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def evaluate(
        self, vehicle_id: str, payload: TwinSampleRequest
    ) -> TwinFrame:
        return await self._evaluate(vehicle_id, payload)

    async def evaluate_observation(
        self,
        vehicle_id: str,
        payload: TwinSampleRequest,
        thermal_image: bytes,
    ) -> TwinFrame:
        return await self._evaluate(vehicle_id, payload, thermal_image=thermal_image)

    async def _evaluate(
        self,
        vehicle_id: str,
        payload: TwinSampleRequest,
        *,
        thermal_image: bytes | None = None,
    ) -> TwinFrame:
        validate_vehicle_id(vehicle_id)
        lock = await self._vehicle_lock(vehicle_id)
        async with lock:
            latest = await self.redis.get_latest(vehicle_id)
            session_changed = (
                latest is not None and payload.session_id != latest.session_id
            )
            if session_changed:
                await self.current_stage.reset(vehicle_id)
            elif latest is not None:
                if payload.sequence != latest.sequence + 1:
                    raise TwinSequenceConflict(
                        "sequence must advance by exactly one at 1 Hz"
                    )
                elapsed = (payload.observed_at - latest.observed_at).total_seconds()
                if abs(elapsed - 1.0) >= 1e-6:
                    raise TwinSequenceConflict(
                        "observed_at must advance by exactly one second at 1 Hz"
                    )
            model_request = _model_request(payload)
            result = await self.current_stage.evaluate(vehicle_id, model_request)
            ml_level = (
                None
                if result.ml_pattern_stage is None
                else _STAGE_LEVEL[result.ml_pattern_stage]
            )
            physics_level = _STAGE_LEVEL[result.physical_rule_level]
            bms_final_level = _STAGE_LEVEL[result.final_safety_alert]
            thermal_result = None
            thermal_cell_heat_score = None
            module_heat_score = None
            hotspot_module_index = None
            thermal_frame_sha256 = None
            if thermal_image is not None:
                thermal_cell_heat_score, _ = analyze_cell_heat_scores(
                    thermal_image
                )
                thermal_frame_sha256 = hashlib.sha256(thermal_image).hexdigest()
                thermal_result = await self.thermal_inference.infer(
                    vehicle_id=vehicle_id,
                    observed_at=payload.observed_at.isoformat(),
                    sequence=payload.sequence,
                    layout_id=payload.layout_id,
                    image_bytes=thermal_image,
                )
            fused = fuse_twin_state(
                payload,
                ml_level=ml_level,
                physical_rule_level=physics_level,
                bms_final_level=bms_final_level,
                thermal_result=thermal_result,
                thermal_cell_heat_score=thermal_cell_heat_score,
                module_heat_score=module_heat_score,
                hotspot_module_index=hotspot_module_index,
                thermal_frame_sha256=thermal_frame_sha256,
                minimum_image_confidence=self.minimum_image_confidence,
            )
            frame = TwinFrame(
                vehicle_id=vehicle_id,
                session_id=payload.session_id,
                observed_at=payload.observed_at,
                sequence=payload.sequence,
                temperature_decic=list(payload.temperature_decic),
                voltage_mv=list(payload.voltage_mv),
                state_level=fused["state_level"],
                connector_temperature_decic=list(
                    payload.connector_temperature_decic
                ),
                connector_state_level=fused["connector_state_level"],
                hotspot_cell_index=fused["hotspot_cell_index"],
                hotspot_connector_index=max(
                    range(len(payload.connector_temperature_decic)),
                    key=payload.connector_temperature_decic.__getitem__,
                ),
                ml_risk_level=fused.get("ml_risk_level", ml_level),
                physics_risk_level=fused["physics_risk_level"],
                final_risk_level=fused["final_risk_level"],
                image_risk_level=fused["image_risk_level"],
                image_confidence=fused["image_confidence"],
                image_probabilities=fused["image_probabilities"],
                image_model_status=fused["image_model_status"],
                cell_heat_score=fused["cell_heat_score"],
                module_heat_score=fused["module_heat_score"],
                module_state_level=fused["module_state_level"],
                hotspot_module_index=fused["hotspot_module_index"],
                thermal_frame_sha256=fused["thermal_frame_sha256"],
                fusion_source=fused["fusion_source"],
            )
            if self.anomaly_persistence is not None and frame.final_risk_level > 0:
                persistence_payload = model_request.model_copy(
                    update={
                        "hotspot_cell_index": frame.hotspot_cell_index,
                        "hotspot_connector_index": frame.hotspot_connector_index,
                        "image_risk_level": frame.image_risk_level,
                        "image_confidence": frame.image_confidence,
                        "source_image_ref": frame.thermal_frame_ref,
                    }
                )
                persistence_result = result.model_copy(
                    update={
                        "physical_rule_level": _LEVEL_STAGE[frame.physics_risk_level],
                        "final_safety_alert": _LEVEL_STAGE[frame.final_risk_level],
                    }
                )
                anomaly_id = await self.anomaly_persistence.persist_if_anomalous(
                    vehicle_id,
                    persistence_payload,
                    persistence_result,
                    frame,
                )
                if anomaly_id is not None:
                    frame = frame.model_copy(update={"anomaly_id": anomaly_id})
            await self.redis.publish(frame)
            return frame

    async def risk_vehicles(self) -> RiskVehicleListResponse:
        return RiskVehicleListResponse(items=await self.redis.risk_vehicles())

    async def latest(self, vehicle_id: str) -> TwinFrame:
        """Return only the live Redis snapshot, never an old incident frame."""

        validate_vehicle_id(vehicle_id)
        frame = await self.redis.get_latest(vehicle_id)
        if frame is None:
            raise IncidentNotFound("no live vehicle state is available")
        return frame

    async def incidents(self, vehicle_id: str) -> IncidentListResponse:
        validate_vehicle_id(vehicle_id)
        async with self.sessions() as session:
            return IncidentListResponse(
                items=await self.repository.list_incidents(session, vehicle_id)
            )

    async def latest_history(
        self, vehicle_id: str, resolution_seconds: int
    ) -> TwinHistoryResponse:
        validate_vehicle_id(vehicle_id)
        if not 1 <= resolution_seconds <= 3_600:
            raise ValueError("resolution_seconds must be between 1 and 3600")
        async with self.sessions() as session:
            incident = await self.repository.latest_complete_incident(
                session, vehicle_id
            )
            if incident is None:
                raise IncidentNotFound(
                    "no completed incident history exists for this vehicle"
                )
            return await self._history_response(
                session, incident, resolution_seconds
            )

    async def incident_history(
        self,
        vehicle_id: str,
        incident_id: str,
        resolution_seconds: int,
    ) -> TwinHistoryResponse:
        validate_vehicle_id(vehicle_id)
        if not 1 <= resolution_seconds <= 3_600:
            raise ValueError("resolution_seconds must be between 1 and 3600")
        async with self.sessions() as session:
            incident = await self.repository.complete_incident(
                session, vehicle_id, incident_id
            )
            if incident is None:
                raise IncidentNotFound(
                    "no completed incident history exists for this vehicle and incident"
                )
            return await self._history_response(
                session, incident, resolution_seconds
            )

    async def _history_response(
        self,
        session: AsyncSession,
        incident,
        resolution_seconds: int,
    ) -> TwinHistoryResponse:
        summary = self.repository._summary(
            incident,
            await self.repository.count_frames(session, incident.id),
        )
        frames = await self.repository.history(
            session, incident, resolution_seconds
        )
        return TwinHistoryResponse(
            incident=summary,
            resolution_seconds=resolution_seconds,
            frames=frames,
        )

    async def live_frames(self, vehicle_id: str) -> AsyncIterator[TwinFrame]:
        """Subscribe first, then read latest, and suppress the race duplicate by sequence."""

        validate_vehicle_id(vehicle_id)
        pubsub = await self.redis.subscribe(vehicle_id)
        last_sequence = -1
        try:
            latest = await self.redis.get_latest(vehicle_id)
            if latest is not None:
                last_sequence = latest.sequence
                yield latest
            async for frame in self.redis.live_messages(pubsub):
                if frame.sequence <= last_sequence:
                    continue
                last_sequence = frame.sequence
                yield frame
        finally:
            await self.redis.close_subscription(pubsub)

    async def _vehicle_lock(self, vehicle_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(vehicle_id, asyncio.Lock())
