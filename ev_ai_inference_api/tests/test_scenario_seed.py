from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.scenario_catalog import SCENARIO_BY_ID
from app.scenario_generator import (
    HISTORY_PRE_SECONDS,
    generate_scenario_frames,
    write_scenario_dataset,
)
from app.scenario_seed import plan_history_seed, seed_history


START = datetime(2026, 8, 7, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_plan_history_seed_only_anomaly_scenarios(tmp_path) -> None:
    for scenario_id in ("normal", "connector_local_overheat"):
        scenario = SCENARIO_BY_ID[scenario_id]
        frames = await generate_scenario_frames(
            scenario,
            frame_count=5,
            start_at=START,
        )
        write_scenario_dataset(scenario, frames, tmp_path)

    plans = plan_history_seed(tmp_path)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.scenario_id == "connector_local_overheat"
    assert plan.vehicle_id == "scenario-connector_local_overheat"
    assert plan.incident_type == "connector"
    assert plan.window_start == START
    assert plan.triggered_at == START + timedelta(seconds=HISTORY_PRE_SECONDS)
    assert plan.window_end == START + timedelta(seconds=5)
    assert plan.frame_count == 5
    assert plan.incident_id


@pytest.mark.asyncio
async def test_seed_history_dry_run_returns_plan(tmp_path) -> None:
    scenario = SCENARIO_BY_ID["battery_over_temp"]
    frames = await generate_scenario_frames(
        scenario,
        frame_count=5,
        start_at=START,
    )
    write_scenario_dataset(scenario, frames, tmp_path)

    plans = await seed_history(
        tmp_path,
        database_url="postgresql+asyncpg://unused",
        dry_run=True,
    )
    assert len(plans) == 1
    assert plans[0].scenario_id == "battery_over_temp"
