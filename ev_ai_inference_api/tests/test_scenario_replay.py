from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from app.scenario_catalog import SCENARIO_BY_ID
from app.scenario_generator import (
    generate_scenario_frames,
    write_scenario_dataset,
)
from app.scenario_replay import (
    VehicleScenarioAssignment,
    load_compact_datasets,
    load_assignments_from_file,
    replay_scenarios,
)


START = datetime(2026, 8, 7, tzinfo=timezone.utc)


class FakeStore:
    def __init__(self) -> None:
        self.published: list = []

    async def publish_live_only(self, frame) -> None:
        self.published.append(frame)


def test_load_assignments_from_file(tmp_path) -> None:
    path = tmp_path / "assignments.json"
    path.write_text(
        json.dumps(
            [
                {
                    "vehicle_id": "11111111-1111-1111-1111-111111111111",
                    "car_number": "11가6762",
                    "model": "GV60",
                    "scenario_id": "connector_local_overheat",
                    "offset_seconds": 0,
                },
                {
                    "vehicle_id": "22222222-2222-2222-2222-222222222222",
                    "car_number": "14가7638",
                    "model": "아이오닉6",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assignments = load_assignments_from_file(path)
    assert len(assignments) == 2
    assert assignments[0].scenario_id == "connector_local_overheat"
    assert assignments[1].scenario_id == "normal"
    assert assignments[1].offset_seconds == 1


def test_load_assignments_rejects_unknown_scenario(tmp_path) -> None:
    path = tmp_path / "assignments.json"
    path.write_text(
        json.dumps(
            [
                {
                    "vehicle_id": "car-1",
                    "scenario_id": "does_not_exist",
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown scenario_id"):
        load_assignments_from_file(path)


@pytest.mark.asyncio
async def test_load_datasets_reads_generated_scenarios(tmp_path) -> None:
    scenario = SCENARIO_BY_ID["normal"]
    frames = await generate_scenario_frames(
        scenario,
        frame_count=5,
        start_at=START,
    )
    write_scenario_dataset(scenario, frames, tmp_path)
    datasets = load_compact_datasets(tmp_path)
    assert datasets["normal"]
    assert datasets["normal"].frame_count == 5


@pytest.mark.asyncio
async def test_replay_scenarios_publishes_rewritten_frames(tmp_path) -> None:
    scenario = SCENARIO_BY_ID["normal"]
    frames = await generate_scenario_frames(
        scenario,
        frame_count=10,
        start_at=START,
    )
    write_scenario_dataset(scenario, frames, tmp_path)
    datasets = load_compact_datasets(tmp_path)
    assignments = [
        VehicleScenarioAssignment(
            vehicle_id="car-11111111-1111-1111-1111-111111111111",
            car_number="11가6762",
            model="GV60",
            scenario_id="normal",
            offset_seconds=0,
        ),
        VehicleScenarioAssignment(
            vehicle_id="car-22222222-2222-2222-2222-222222222222",
            car_number="14가7638",
            model="아이오닉6",
            scenario_id="normal",
            offset_seconds=2,
        ),
    ]
    store = FakeStore()
    count = await replay_scenarios(
        store,
        assignments,
        datasets,
        start_at=START,
        speed=1_000.0,
        duration_seconds=3,
    )
    assert count == 3
    assert len(store.published) == 6
    first_car = [
        frame for frame in store.published if frame.vehicle_id == assignments[0].vehicle_id
    ]
    second_car = [
        frame for frame in store.published if frame.vehicle_id == assignments[1].vehicle_id
    ]
    assert [frame.sequence for frame in first_car] == [0, 1, 2]
    assert [frame.observed_at for frame in first_car] == [
        START + timedelta(seconds=index)
        for index in range(3)
    ]
    assert first_car[0].temperature_decic == frames[0].temperature_decic
    assert second_car[0].temperature_decic == frames[2].temperature_decic
