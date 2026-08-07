from __future__ import annotations
from typing import Any
from app.core.session_manager import SessionManager
from app.schemas.current_stage import CurrentStageResponse, SampleRequest

class CurrentStageService:
    def __init__(self, sessions: SessionManager) -> None: self.sessions = sessions
    async def evaluate(self, vehicle_id: str, request: SampleRequest) -> CurrentStageResponse:
        payload = request.model_dump()
        result = await self.sessions.push(vehicle_id, payload, request.timestamp_seconds)
        route = "warming_up" if result.ml_pattern_stage is None else ("stage_120s" if result.history_seconds >= 120 else "stage_30s")
        return CurrentStageResponse(vehicle_id=vehicle_id, sensor_health=result.sensor_health, history_seconds=result.history_seconds, model_route=route, ml_pattern_stage=result.ml_pattern_stage, current_stage_probabilities=result.ml_probabilities, physical_rule_level=result.physical_rule_level, final_safety_alert=result.final_safety_alert, charging_equipment_observation=result.charging_equipment_observation, reason_codes=result.reason_codes)
    async def reset(self, vehicle_id: str) -> dict[str, Any]: await self.sessions.reset(vehicle_id); return {"vehicle_id": vehicle_id, "reset": True}
    async def delete(self, vehicle_id: str) -> dict[str, Any]: return {"vehicle_id": vehicle_id, "deleted": await self.sessions.delete(vehicle_id)}
