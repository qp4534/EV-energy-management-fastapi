from datetime import datetime, timezone

import pytest

from app.db.models import INCIDENT_TYPE_BATTERY, INCIDENT_TYPE_CONNECTOR
from app.demo_profiles import load_demo_profiles, profile_frame_at
from app.simulator import (
    KOREA_TIMEZONE,
    SEED_FRAME_COUNT,
    SEED_TRIGGER_FRAME_INDEX,
    VEHICLE_PROFILES,
    _cap_profile_frame,
    _incident_trigger_at,
    build_seed_frame,
)


START = datetime(2026, 7, 31, tzinfo=timezone.utc)


def test_vehicle_profiles_match_controller_car_list() -> None:
    assert [profile.vehicle_id for profile in VEHICLE_PROFILES] == [
        f"car-uuid-{index:03d}" for index in range(1, 11)
    ]
    assert [profile.risk_level for profile in VEHICLE_PROFILES] == [
        3, 2, 1, 0, 2, 0, 3, 0, 1, 0
    ]
    assert [profile.abnormal_type for profile in VEHICLE_PROFILES] == [
        "temperature_rise",
        "fire_risk",
        "temperature_rise",
        "normal",
        "overcharge_warning",
        "normal",
        "fire_risk",
        "normal",
        "temperature_rise",
        "normal",
    ]
    assert [profile.charging_time.isoformat() for profile in VEHICLE_PROFILES] == [
        "14:00:00",
        "14:00:00",
        "14:00:00",
        "14:12:30",
        "14:25:10",
        "14:30:00",
        "14:40:15",
        "14:45:00",
        "15:00:00",
        "15:10:20",
    ]


def test_profile_charging_time_is_the_incident_trigger_in_korea() -> None:
    for profile in VEHICLE_PROFILES:
        trigger = _incident_trigger_at(START, profile)
        assert trigger.astimezone(KOREA_TIMEZONE).time() == profile.charging_time


def test_profile_risk_cap_preserves_shape_without_exceeding_car_stage() -> None:
    profiles = load_demo_profiles()
    for profile in VEHICLE_PROFILES:
        if profile.incident_type is None:
            continue
        source = profiles[profile.incident_type].frames[-1]
        capped = _cap_profile_frame(source, profile.risk_level)
        if capped.state_level is not None:
            assert max(capped.state_level) <= profile.risk_level
        if capped.connector_state_level is not None:
            assert max(capped.connector_state_level) <= profile.risk_level


def _seed_frame(incident_type: str, source_index: int):
    source = load_demo_profiles()[incident_type].frames[source_index]
    return build_seed_frame(
        vehicle_id="car-uuid-002",
        vehicle_number=1,
        observed_at=START,
        sequence=3_600,
        risk_level=source.risk_level,
        incident_type=incident_type,
        profile_frame=source,
    )


def test_original_demo_profile_keeps_natural_spatial_heat_gradient() -> None:
    connector = _seed_frame(INCIDENT_TYPE_CONNECTOR, 90)
    assert connector.connector_temperature_decic == [868, 731, 481]
    assert connector.connector_state_level == [3, 2, 1]
    assert max(connector.state_level) == 0
    assert connector.ml_risk_level == 0
    assert connector.final_risk_level == 3

    battery = _seed_frame(INCIDENT_TYPE_BATTERY, 100)
    assert max(battery.connector_state_level) == 0
    assert battery.hotspot_cell_index == 51
    assert battery.state_level.count(3) == 22
    assert battery.state_level.count(2) == 74
    assert battery.temperature_decic[48:64] == [
        768,
        803,
        831,
        845,
        764,
        797,
        821,
        827,
        810,
        781,
        748,
        725,
        803,
        775,
        744,
        724,
    ]
    assert battery.ml_risk_level == 3
    assert battery.final_risk_level == 3


@pytest.mark.parametrize(
    "incident_type",
    [INCIDENT_TYPE_CONNECTOR, INCIDENT_TYPE_BATTERY],
)
def test_three_hour_seed_maps_original_profile_around_incident_trigger(
    incident_type: str,
) -> None:
    scenario = load_demo_profiles()[incident_type]
    assert profile_frame_at(
        scenario,
        0,
        output_frame_count=SEED_FRAME_COUNT,
        trigger_frame_index=SEED_TRIGGER_FRAME_INDEX,
    ) is scenario.frames[0]
    assert profile_frame_at(
        scenario,
        SEED_TRIGGER_FRAME_INDEX,
        output_frame_count=SEED_FRAME_COUNT,
        trigger_frame_index=SEED_TRIGGER_FRAME_INDEX,
    ) is scenario.frames[scenario.first_visual_risk_frame]
    assert profile_frame_at(
        scenario,
        SEED_FRAME_COUNT - 1,
        output_frame_count=SEED_FRAME_COUNT,
        trigger_frame_index=SEED_TRIGGER_FRAME_INDEX,
    ) is scenario.frames[-1]


def test_typed_seed_frame_cannot_fall_back_to_flat_red_demo_data() -> None:
    with pytest.raises(ValueError, match="original demo profile"):
        build_seed_frame(
            vehicle_id="car-uuid-002",
            vehicle_number=1,
            observed_at=START,
            sequence=3_600,
            risk_level=3,
            incident_type=INCIDENT_TYPE_BATTERY,
        )
