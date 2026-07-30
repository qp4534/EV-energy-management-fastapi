from __future__ import annotations
from math import isfinite
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .common import Stage

class SampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp_seconds: float
    voltage_v: float
    temp_mean_c: float
    temp_max_c: float
    temp_delta_c: float
    temp_saturation_fraction: float
    temp_saturation_all: bool
    raw_temp_max_c: float | None = None
    raw_temp_mean_c: float | None = None
    ambient_temp_c: float | None = None
    pack_current_a: float | None = None
    cell_voltages_v: list[float] | None = None
    charging_gun_temperature_c: float | None = None

    @field_validator("timestamp_seconds", "voltage_v", "temp_mean_c", "temp_max_c", "temp_delta_c", "temp_saturation_fraction", "raw_temp_max_c", "raw_temp_mean_c", "ambient_temp_c", "pack_current_a", "charging_gun_temperature_c")
    @classmethod
    def finite(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value): raise ValueError("must be a finite number")
        return value
    @field_validator("cell_voltages_v")
    @classmethod
    def finite_cells(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and not all(isfinite(x) for x in value): raise ValueError("cell_voltages_v must contain finite numbers")
        return value

class CurrentStageResponse(BaseModel):
    vehicle_id: str; sensor_health: Literal["good", "invalid"]; history_seconds: int
    model_route: Literal["warming_up", "stage_30s", "stage_120s"]
    ml_pattern_stage: Stage | None = None
    current_stage_probabilities: dict[Stage, float] | None = None
    physical_rule_level: Stage; final_safety_alert: Stage | Literal["unknown"]
    charging_equipment_observation: str; reason_codes: list[str]
