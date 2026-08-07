from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.scenario_catalog import SCENARIO_BY_ID
from app.scenario_generator import (
    HISTORY_FRAME_COUNT,
    SCENARIO_FRAME_COUNT,
    build_scenario_sample,
    generate_scenario_frames,
    load_scenario_frames,
    write_scenario_dataset,
)


START = datetime(2026, 8, 7, tzinfo=timezone.utc)


def test_build_scenario_sample_normal_is_stable() -> None:
    scenario = SCENARIO_BY_ID["normal"]
    frame = build_scenario_sample(
        scenario,
        1_800,
        frame_count=SCENARIO_FRAME_COUNT,
    )
    assert max(frame.temperature_decic) < 350
    assert min(frame.temperature_decic) > 290
    assert max(frame.voltage_mv) < 3_900
    assert max(frame.connector_temperature_decic) < 330


@pytest.mark.parametrize(
    ("scenario_id", "check"),
    [
        (
            "connector_local_overheat",
            lambda frame: frame.connector_temperature_decic[0] >= 800,
        ),
        (
            "battery_over_temp",
            lambda frame: max(frame.temperature_decic) >= 800
            and min(frame.voltage_mv) <= 2_000,
        ),
        (
            "thermal_runaway_risk",
            lambda frame: max(frame.temperature_decic) >= 800,
        ),
        (
            "cell_voltage_imbalance",
            lambda frame: frame.voltage_mv[51] <= 2_000,
        ),
        (
            "battery_overheat_sign",
            lambda frame: max(frame.temperature_decic) >= 600,
        ),
        (
            "rapid_temp_rise",
            lambda frame: max(frame.temperature_decic) >= 600,
        ),
        (
            "connector_temp_rise",
            lambda frame: frame.connector_temperature_decic[0] >= 450,
        ),
        (
            "cell_voltage_deviation",
            lambda frame: frame.voltage_mv[51] <= 2_400,
        ),
    ],
)
def test_build_scenario_sample_reaches_scenario_signal(
    scenario_id: str,
    check,
) -> None:
    scenario = SCENARIO_BY_ID[scenario_id]
    frame = build_scenario_sample(
        scenario,
        SCENARIO_FRAME_COUNT - 1,
        frame_count=SCENARIO_FRAME_COUNT,
    )
    assert check(frame)


def test_charging_current_fluctuation_changes_pack_current() -> None:
    scenario = SCENARIO_BY_ID["charging_current_fluctuation"]
    samples = [
        build_scenario_sample(scenario, index, frame_count=SCENARIO_FRAME_COUNT)
        for index in (0, 5, 17)
    ]
    currents = {sample.pack_current_a for sample in samples}
    assert len(currents) > 1
    assert max(currents) - min(currents) > 20


def test_live_plateau_keeps_anomaly_with_heat_spread() -> None:
    scenario = SCENARIO_BY_ID["battery_over_temp"]
    frame = build_scenario_sample(
        scenario,
        0,
        frame_count=SCENARIO_FRAME_COUNT,
        anomaly_plateau=True,
    )
    assert max(frame.temperature_decic) >= 800
    assert frame.temperature_decic[51] > frame.temperature_decic[30]
    assert frame.temperature_decic[50] > 400


def test_live_plateau_values_are_not_identical() -> None:
    scenario = SCENARIO_BY_ID["battery_over_temp"]
    first = build_scenario_sample(
        scenario,
        0,
        frame_count=SCENARIO_FRAME_COUNT,
        anomaly_plateau=True,
    )
    later = build_scenario_sample(
        scenario,
        300,
        frame_count=SCENARIO_FRAME_COUNT,
        anomaly_plateau=True,
    )
    assert sum(first.temperature_decic) != sum(later.temperature_decic)


def test_connector_plateau_visible_from_start() -> None:
    scenario = SCENARIO_BY_ID["connector_local_overheat"]
    frame = build_scenario_sample(
        scenario,
        0,
        frame_count=SCENARIO_FRAME_COUNT,
        anomaly_plateau=True,
    )
    assert frame.connector_temperature_decic[0] >= 800


def test_history_progressive_spreads_heat_to_neighbors() -> None:
    scenario = SCENARIO_BY_ID["battery_over_temp"]
    frame = build_scenario_sample(
        scenario,
        HISTORY_FRAME_COUNT - 1,
        frame_count=HISTORY_FRAME_COUNT,
    )
    assert max(frame.temperature_decic) >= 800
    assert frame.temperature_decic[51] > frame.temperature_decic[30]
    assert frame.temperature_decic[50] > 400


def test_thermal_runaway_all_cells_fluctuate() -> None:
    scenario = SCENARIO_BY_ID["thermal_runaway_risk"]
    first = build_scenario_sample(
        scenario,
        0,
        frame_count=SCENARIO_FRAME_COUNT,
        anomaly_plateau=True,
    )
    later = build_scenario_sample(
        scenario,
        100,
        frame_count=SCENARIO_FRAME_COUNT,
        anomaly_plateau=True,
    )
    changed_cells = sum(
        left != right
        for left, right in zip(
            first.temperature_decic,
            later.temperature_decic,
            strict=True,
        )
    )
    assert changed_cells >= 20
    assert sum(first.temperature_decic) != sum(later.temperature_decic)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario_id",
    sorted(SCENARIO_BY_ID),
)
async def test_generate_scenario_frames_matches_target_risk(scenario_id: str) -> None:
    scenario = SCENARIO_BY_ID[scenario_id]
    frames = await generate_scenario_frames(
        scenario,
        frame_count=5,
        start_at=START,
        settings=None,
    )
    assert len(frames) == 5
    assert [frame.sequence for frame in frames] == list(range(5))
    assert frames[-1].final_risk_level == scenario.risk_level
    assert all(
        frame.final_risk_level == scenario.risk_level for frame in frames
    )
    assert frames[-1].vehicle_id == f"scenario-{scenario.scenario_id}"


@pytest.mark.asyncio
async def test_dataset_write_and_load_round_trip(tmp_path) -> None:
    scenario = SCENARIO_BY_ID["normal"]
    frames = await generate_scenario_frames(
        scenario,
        frame_count=10,
        start_at=START,
    )
    scenario_dir = write_scenario_dataset(scenario, frames, tmp_path)
    frames_path = scenario_dir / "frames.jsonl.gz"
    assert frames_path.is_file()
    assert (scenario_dir / "metadata.json").is_file()
    loaded = load_scenario_frames(frames_path)
    assert len(loaded) == 10
    assert loaded[0].model_dump() == frames[0].model_dump()
