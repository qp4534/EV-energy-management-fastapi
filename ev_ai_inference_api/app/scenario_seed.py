from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.core.config import Settings
from app.db.models import INCIDENT_TYPE_GENERAL
from app.db.repository import TwinRepository
from app.db.session import create_database
from app.scenario_catalog import SCENARIO_BY_ID
from app.scenario_generator import (
    HISTORY_FRAME_COUNT,
    HISTORY_PRE_SECONDS,
    load_scenario_frames,
)
from app.scenario_replay import load_datasets
from app.workers.twin_persistence import deterministic_incident_id


@dataclass(frozen=True)
class HistorySeedPlan:
    scenario_id: str
    vehicle_id: str
    incident_id: str
    incident_type: str
    window_start: datetime
    triggered_at: datetime
    window_end: datetime
    frame_count: int


def plan_history_seed(scenario_dir: Path) -> list[HistorySeedPlan]:
    """Build incident plans for every generated anomaly scenario dataset."""

    datasets = load_datasets(scenario_dir)
    plans: list[HistorySeedPlan] = []
    for scenario_id, frames in sorted(datasets.items()):
        scenario = SCENARIO_BY_ID[scenario_id]
        if scenario.risk_level == 0:
            continue
        if not frames:
            raise ValueError(f"{scenario_id} has no frames")
        window_start = frames[0].observed_at
        frame_count = len(frames)
        triggered_at = window_start + timedelta(seconds=HISTORY_PRE_SECONDS)
        window_end = window_start + timedelta(seconds=frame_count)
        vehicle_id = f"scenario-{scenario_id}"
        plans.append(
            HistorySeedPlan(
                scenario_id=scenario_id,
                vehicle_id=vehicle_id,
                incident_id=deterministic_incident_id(
                    vehicle_id, triggered_at
                ),
                incident_type=scenario.incident_type or INCIDENT_TYPE_GENERAL,
                window_start=window_start,
                triggered_at=triggered_at,
                window_end=window_end,
                frame_count=frame_count,
            )
        )
    return plans


async def seed_history(
    scenario_dir: Path,
    *,
    database_url: str,
    dry_run: bool = False,
) -> list[HistorySeedPlan]:
    plans = plan_history_seed(scenario_dir)
    if not plans:
        raise SystemExit("no anomaly scenario datasets found to seed")
    datasets = load_datasets(scenario_dir)

    if dry_run:
        for plan in plans:
            print(
                f"[dry-run] {plan.scenario_id}: vehicle={plan.vehicle_id} "
                f"incident={plan.incident_id} type={plan.incident_type} "
                f"frames={plan.frame_count}"
            )
        return plans

    engine, sessions = create_database(database_url)
    repository = TwinRepository()
    try:
        for plan in plans:
            frames = datasets[plan.scenario_id]
            if len(frames) != HISTORY_FRAME_COUNT:
                raise ValueError(
                    f"{plan.scenario_id} must contain {HISTORY_FRAME_COUNT} "
                    f"frames, got {len(frames)}"
                )
            async with sessions() as session:
                async with session.begin():
                    await repository.create_incident(
                        session,
                        incident_id=plan.incident_id,
                        vehicle_id=plan.vehicle_id,
                        triggered_at=plan.triggered_at,
                        window_start=plan.window_start,
                        window_end=plan.window_end,
                        incident_type=plan.incident_type,
                    )
                    await repository.replace_seed_frames(
                        session,
                        plan.incident_id,
                        frames,
                    )
                    await repository.mark_complete(
                        session,
                        plan.incident_id,
                        plan.window_end,
                    )
            print(
                f"seeded {plan.scenario_id}: vehicle={plan.vehicle_id} "
                f"incident={plan.incident_id} frames={len(frames)}"
            )
    finally:
        await engine.dispose()
    return plans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed 사고 전후 3시간 incident histories for anomaly scenarios"
    )
    parser.add_argument(
        "--scenario-dir",
        default=str(Path.cwd() / "runtime" / "scenarios-history"),
        help="directory containing generated history datasets",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="SQLAlchemy database URL; defaults to DATABASE_URL env",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the seed plan without writing to the database",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database_url = args.db_url or Settings.load().database_url
    asyncio.run(
        seed_history(
            Path(args.scenario_dir),
            database_url=database_url,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
