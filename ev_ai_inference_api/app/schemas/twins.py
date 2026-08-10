from __future__ import annotations

from datetime import datetime
import math
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator


TWIN_SCHEMA_VERSION = 1
TWIN_LAYOUT_ID = "generic_ev_concept_96_v1"
CELL_COUNT = 96
CONNECTOR_COMPONENT_COUNT = 3
MODULE_COUNT = 12

SmallInt = Annotated[int, Field(strict=True, ge=-32_768, le=32_767)]
StateLevel = Annotated[int, Field(strict=True, ge=0, le=3)]
SequenceNumber = Annotated[int, Field(strict=True, ge=0)]


def _require_length(values: list[int], expected: int, field_name: str) -> list[int]:
    if len(values) != expected:
        raise ValueError(f"{field_name} must contain exactly {expected} values")
    return values


class TwinSampleRequest(BaseModel):
    """One logical 1 Hz sensor observation used to produce a public TwinFrame."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = TWIN_SCHEMA_VERSION
    layout_id: Literal["generic_ev_concept_96_v1"] = TWIN_LAYOUT_ID
    session_id: UUID | None = None
    observed_at: AwareDatetime
    sequence: SequenceNumber
    temperature_decic: list[SmallInt]
    voltage_mv: list[SmallInt]
    connector_temperature_decic: list[SmallInt]
    ambient_temperature_c: float | None = None
    pack_current_a: float | None = None

    @field_validator("temperature_decic")
    @classmethod
    def validate_temperatures(cls, values: list[int]) -> list[int]:
        values = _require_length(values, CELL_COUNT, "temperature_decic")
        if any(value < -400 or value > 1_500 for value in values):
            raise ValueError("temperature_decic values must be between -400 and 1500")
        return values

    @field_validator("voltage_mv")
    @classmethod
    def validate_voltages(cls, values: list[int]) -> list[int]:
        values = _require_length(values, CELL_COUNT, "voltage_mv")
        if any(value < 0 or value > 6_000 for value in values):
            raise ValueError("voltage_mv values must be between 0 and 6000")
        return values

    @field_validator("connector_temperature_decic")
    @classmethod
    def validate_connector_temperatures(cls, values: list[int]) -> list[int]:
        return _require_length(
            values,
            CONNECTOR_COMPONENT_COUNT,
            "connector_temperature_decic",
        )


class ThermalInferenceResult(BaseModel):
    """Normalized result returned by the optional local thermal model worker."""

    model_id: str | None = None
    label_scheme: Literal["stage4_1_23_45_6"] = "stage4_1_23_45_6"
    risk_level: StateLevel | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    probabilities: list[float] | None = None
    status: Literal["ready", "unavailable", "unaligned", "unqualified"] = "unavailable"

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, values: list[float] | None) -> list[float] | None:
        if values is None:
            return None
        if len(values) != 4 or any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("probabilities must contain four values between 0 and 1")
        if abs(sum(values) - 1.0) > 1e-4:
            raise ValueError("probabilities must sum to 1")
        return values


class TwinFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = TWIN_SCHEMA_VERSION
    layout_id: Literal["generic_ev_concept_96_v1"] = TWIN_LAYOUT_ID
    vehicle_id: str
    anomaly_id: str | None = None
    session_id: UUID | None = None
    observed_at: AwareDatetime
    sequence: SequenceNumber
    temperature_decic: list[SmallInt]
    voltage_mv: list[SmallInt]
    state_level: list[StateLevel]
    connector_temperature_decic: list[SmallInt]
    connector_state_level: list[StateLevel]
    hotspot_cell_index: Annotated[int, Field(strict=True, ge=0, lt=CELL_COUNT)]
    hotspot_connector_index: Annotated[
        int, Field(strict=True, ge=0, lt=CONNECTOR_COMPONENT_COUNT)
    ]
    ml_risk_level: StateLevel | None
    physics_risk_level: StateLevel
    final_risk_level: StateLevel
    cell_heat_score: list[float] | None = None
    image_risk_level: StateLevel | None = None
    image_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    image_probabilities: list[float] | None = None
    image_model_status: Literal[
        "ready", "unavailable", "unaligned", "unqualified"
    ] = "unavailable"
    module_heat_score: list[float] | None = None
    module_state_level: list[StateLevel] | None = None
    hotspot_module_index: int | None = Field(default=None, ge=0, lt=MODULE_COUNT)
    thermal_frame_ref: str | None = None
    thermal_frame_sha256: str | None = None
    fusion_source: Literal[
        "sensor-only", "image+sensor", "physics", "image-unqualified"
    ] = "sensor-only"

    @field_validator("temperature_decic", "voltage_mv", "state_level")
    @classmethod
    def validate_cells(cls, values: list[int], info) -> list[int]:
        return _require_length(values, CELL_COUNT, info.field_name)

    @field_validator("connector_temperature_decic", "connector_state_level")
    @classmethod
    def validate_connector(cls, values: list[int], info) -> list[int]:
        return _require_length(values, CONNECTOR_COMPONENT_COUNT, info.field_name)

    @field_validator("cell_heat_score")
    @classmethod
    def validate_cell_heat_score(
        cls, values: list[float] | None
    ) -> list[float] | None:
        if values is None:
            return None
        if len(values) != CELL_COUNT or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in values
        ):
            raise ValueError(
                "cell_heat_score must contain 96 values between 0 and 1"
            )
        return values

    @field_validator("image_probabilities")
    @classmethod
    def validate_image_probabilities(
        cls, values: list[float] | None
    ) -> list[float] | None:
        if values is None:
            return None
        if len(values) != 4 or any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("image_probabilities must contain four values between 0 and 1")
        if abs(sum(values) - 1.0) > 1e-4:
            raise ValueError("image_probabilities must sum to 1")
        return values

    @field_validator("module_heat_score", "module_state_level")
    @classmethod
    def validate_modules(cls, values, info):
        if values is None:
            return None
        return _require_length(values, MODULE_COUNT, info.field_name)


class RiskVehicleItem(BaseModel):
    vehicle_id: str
    observed_at: AwareDatetime
    sequence: SequenceNumber
    final_risk_level: StateLevel


class RiskVehicleListResponse(BaseModel):
    items: list[RiskVehicleItem]


class TwinLatestMeasurement(BaseModel):
    """Small, explicit contract for current vehicle/passport displays."""

    vehicle_id: str
    observed_at: AwareDatetime
    sequence: SequenceNumber
    source: Literal["twin_live"] = "twin_live"
    max_cell_temperature_c: float
    mean_cell_temperature_c: float
    max_connector_temperature_c: float
    min_cell_voltage_v: float
    max_cell_voltage_v: float
    final_risk_level: StateLevel
    age_seconds: float = Field(ge=0.0)
    stale_after_seconds: Annotated[int, Field(strict=True, ge=1, le=300)]
    is_stale: bool


class IncidentSummary(BaseModel):
    id: str
    vehicle_id: str
    incident_type: Literal["general", "connector", "battery"] = "general"
    triggered_at: AwareDatetime
    window_start: AwareDatetime
    window_end: AwareDatetime
    status: Literal["open", "complete", "incomplete"]
    frame_count: int = 0
    rearmed_at: AwareDatetime | None = None


class IncidentListResponse(BaseModel):
    items: list[IncidentSummary]


class TwinHistoryResponse(BaseModel):
    incident: IncidentSummary
    resolution_seconds: Annotated[int, Field(strict=True, ge=1, le=3_600)]
    frames: list[TwinFrame]


def ensure_utc_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observed_at must include a timezone offset")
    return value
