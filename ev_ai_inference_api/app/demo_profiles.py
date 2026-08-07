from __future__ import annotations

import base64
import gzip
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.db.models import INCIDENT_TYPE_BATTERY, INCIDENT_TYPE_CONNECTOR


PROFILE_DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "charging_twin_profiles.json.gz.b64"
)
SOURCE_SCENARIO_BY_INCIDENT = {
    INCIDENT_TYPE_CONNECTOR: "connector_fault",
    INCIDENT_TYPE_BATTERY: "battery_internal",
}


@dataclass(frozen=True)
class DemoProfileFrame:
    risk_level: int
    temperature_decic: tuple[int, ...] | None = None
    voltage_mv: tuple[int, ...] | None = None
    state_level: tuple[int, ...] | None = None
    hotspot_cell_index: int | None = None
    connector_temperature_decic: tuple[int, ...] | None = None
    connector_state_level: tuple[int, ...] | None = None
    hotspot_connector_index: int | None = None


@dataclass(frozen=True)
class DemoScenarioProfile:
    incident_type: str
    source_scenario_id: str
    first_visual_risk_frame: int
    critical_frame: int
    frames: tuple[DemoProfileFrame, ...]


def _integer_sequence(
    value: Any,
    *,
    field: str,
    length: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} items")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{field} must contain integers")
    result = tuple(value)
    if minimum is not None and any(item < minimum for item in result):
        raise ValueError(f"{field} contains a value below {minimum}")
    if maximum is not None and any(item > maximum for item in result):
        raise ValueError(f"{field} contains a value above {maximum}")
    return result


def _integer(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return value


def _decode_payload(path: Path) -> dict[str, Any]:
    encoded = path.read_text(encoding="ascii")
    try:
        payload = gzip.decompress(base64.b64decode(encoded)).decode("utf-8")
        decoded = json.loads(payload)
    except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid charging twin profile: {path}") from error
    if not isinstance(decoded, dict) or decoded.get("schema_version") != 1:
        raise ValueError("unsupported charging twin profile schema")
    return decoded


def _battery_frame(raw: Any, frame_index: int) -> DemoProfileFrame:
    if not isinstance(raw, dict) or not isinstance(raw.get("twin_state"), dict):
        raise ValueError(f"battery frame {frame_index} is malformed")
    state = raw["twin_state"]
    return DemoProfileFrame(
        risk_level=_integer(
            raw.get("risk_level"),
            field=f"battery frame {frame_index} risk_level",
            minimum=0,
            maximum=3,
        ),
        temperature_decic=_integer_sequence(
            state.get("temperature_decic"),
            field=f"battery frame {frame_index} temperature_decic",
            length=96,
        ),
        voltage_mv=_integer_sequence(
            state.get("voltage_mv"),
            field=f"battery frame {frame_index} voltage_mv",
            length=96,
        ),
        state_level=_integer_sequence(
            state.get("state_level"),
            field=f"battery frame {frame_index} state_level",
            length=96,
            minimum=0,
            maximum=3,
        ),
        hotspot_cell_index=_integer(
            state.get("hotspot_cell_index"),
            field=f"battery frame {frame_index} hotspot_cell_index",
            minimum=0,
            maximum=95,
        ),
    )


def _connector_frame(raw: Any, frame_index: int) -> DemoProfileFrame:
    if not isinstance(raw, dict) or not isinstance(raw.get("twin_state"), dict):
        raise ValueError(f"connector frame {frame_index} is malformed")
    state = raw["twin_state"]
    return DemoProfileFrame(
        risk_level=_integer(
            raw.get("risk_level"),
            field=f"connector frame {frame_index} risk_level",
            minimum=0,
            maximum=3,
        ),
        connector_temperature_decic=_integer_sequence(
            state.get("temperature_decic"),
            field=f"connector frame {frame_index} temperature_decic",
            length=3,
        ),
        connector_state_level=_integer_sequence(
            state.get("state_level"),
            field=f"connector frame {frame_index} state_level",
            length=3,
            minimum=0,
            maximum=3,
        ),
        hotspot_connector_index=_integer(
            state.get("hotspot_component_index"),
            field=f"connector frame {frame_index} hotspot_component_index",
            minimum=0,
            maximum=2,
        ),
    )


@lru_cache(maxsize=1)
def load_demo_profiles() -> dict[str, DemoScenarioProfile]:
    payload = _decode_payload(PROFILE_DATA_PATH)
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise ValueError("charging twin profile scenarios must be a list")

    raw_by_id = {
        scenario.get("id"): scenario
        for scenario in raw_scenarios
        if isinstance(scenario, dict)
    }
    profiles: dict[str, DemoScenarioProfile] = {}
    for incident_type, source_id in SOURCE_SCENARIO_BY_INCIDENT.items():
        raw_scenario = raw_by_id.get(source_id)
        if not isinstance(raw_scenario, dict):
            raise ValueError(f"missing source scenario: {source_id}")
        raw_frames = raw_scenario.get("frames")
        if not isinstance(raw_frames, list) or len(raw_frames) < 2:
            raise ValueError(f"source scenario {source_id} has no usable frames")
        parser = (
            _battery_frame
            if incident_type == INCIDENT_TYPE_BATTERY
            else _connector_frame
        )
        frames = tuple(
            parser(raw_frame, index)
            for index, raw_frame in enumerate(raw_frames)
        )
        last_index = len(frames) - 1
        first_visual = _integer(
            raw_scenario.get("first_visual_risk_frame"),
            field=f"{source_id} first_visual_risk_frame",
            minimum=1,
            maximum=last_index,
        )
        critical = _integer(
            raw_scenario.get("critical_frame"),
            field=f"{source_id} critical_frame",
            minimum=first_visual,
            maximum=last_index,
        )
        profiles[incident_type] = DemoScenarioProfile(
            incident_type=incident_type,
            source_scenario_id=source_id,
            first_visual_risk_frame=first_visual,
            critical_frame=critical,
            frames=frames,
        )
    return profiles


def profile_frame_at(
    scenario: DemoScenarioProfile,
    output_index: int,
    *,
    output_frame_count: int,
    trigger_frame_index: int,
) -> DemoProfileFrame:
    if not 0 <= output_index < output_frame_count:
        raise ValueError("output_index is outside the incident window")
    if not 0 < trigger_frame_index < output_frame_count - 1:
        raise ValueError("trigger_frame_index must be inside the incident window")

    anchor = scenario.first_visual_risk_frame
    source_last = len(scenario.frames) - 1
    if output_index <= trigger_frame_index:
        source_position = anchor * output_index / trigger_frame_index
    else:
        source_position = anchor + (
            (source_last - anchor)
            * (output_index - trigger_frame_index)
            / (output_frame_count - 1 - trigger_frame_index)
        )
    source_index = max(0, min(source_last, round(source_position)))
    return scenario.frames[source_index]
