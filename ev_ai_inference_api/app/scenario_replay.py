from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from redis.asyncio import Redis
from sqlalchemy import text

from app.core.config import Settings
from app.core.twin_redis import TwinRedisStore
from app.db.session import create_database
from app.scenario_catalog import (
    SCENARIOS,
    scenario_for_abnormal_type,
)
from app.scenario_generator import (
    SCENARIO_FRAME_COUNT,
    load_scenario_frames,
)
from app.schemas.twins import TwinFrame


_AWARE_DATETIME = TypeAdapter(datetime)


@dataclass(frozen=True)
class VehicleScenarioAssignment:
    vehicle_id: str
    car_number: str
    model: str
    scenario_id: str
    offset_seconds: int


def _parse_aware(value: str) -> datetime:
    parsed = _AWARE_DATETIME.validate_python(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone offset")
    return parsed


def load_datasets(scenario_dir: Path) -> dict[str, list[TwinFrame]]:
    """Load every available scenario dataset from a generated output directory."""

    datasets: dict[str, list[TwinFrame]] = {}
    for scenario in SCENARIOS:
        frames_path = Path(scenario_dir) / scenario.scenario_id / "frames.jsonl"
        if frames_path.is_file():
            datasets[scenario.scenario_id] = load_scenario_frames(frames_path)
    if not datasets:
        raise ValueError(f"no scenario datasets found under {scenario_dir}")
    return datasets


async def load_vehicle_assignments_from_db(
    sessions,
) -> list[VehicleScenarioAssignment]:
    """Map every CAR to its scenario using the latest ANOMALY_LOGS row."""

    query = text(
        """
        SELECT
            c.car_id::text AS car_id,
            c.car_number,
            c.model,
            latest.abnormal_type
        FROM "CAR" AS c
        LEFT JOIN LATERAL (
            SELECT a."abnormal_type"
            FROM "ANOMALY_LOGS" AS a
            WHERE a."car_id" = c."car_id"
            ORDER BY a."detected_at" DESC
            LIMIT 1
        ) AS latest ON true
        ORDER BY c."car_number"
        """
    )
    async with sessions() as session:
        rows = (await session.execute(query)).mappings().all()

    assignments: list[VehicleScenarioAssignment] = []
    for index, row in enumerate(rows):
        scenario = scenario_for_abnormal_type(row["abnormal_type"])
        assignments.append(
            VehicleScenarioAssignment(
                vehicle_id=row["car_id"],
                car_number=row["car_number"],
                model=row["model"],
                scenario_id=scenario.scenario_id,
                offset_seconds=index % SCENARIO_FRAME_COUNT,
            )
        )
    return assignments


def load_assignments_from_file(path: Path) -> list[VehicleScenarioAssignment]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assignments: list[VehicleScenarioAssignment] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict) or not raw.get("vehicle_id"):
            raise ValueError("each assignment must contain a vehicle_id")
        scenario_id = raw.get("scenario_id", "normal")
        if scenario_id not in {scenario.scenario_id for scenario in SCENARIOS}:
            raise ValueError(f"unknown scenario_id in assignment: {scenario_id}")
        assignments.append(
            VehicleScenarioAssignment(
                vehicle_id=str(raw["vehicle_id"]),
                car_number=str(raw.get("car_number", "")),
                model=str(raw.get("model", "")),
                scenario_id=scenario_id,
                offset_seconds=int(raw.get("offset_seconds", index % SCENARIO_FRAME_COUNT)),
            )
        )
    if not assignments:
        raise ValueError("assignment file contains no vehicles")
    return assignments


async def replay_scenarios(
    store: TwinRedisStore,
    assignments: list[VehicleScenarioAssignment],
    datasets: dict[str, list[TwinFrame]],
    *,
    start_at: datetime,
    speed: float,
    duration_seconds: int | None = None,
) -> int:
    """Publish one pre-computed frame per vehicle per logical second."""

    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be a positive finite number")
    missing = {
        assignment.scenario_id
        for assignment in assignments
        if assignment.scenario_id not in datasets
    }
    if missing:
        raise ValueError(f"missing scenario datasets: {sorted(missing)}")

    loop_second = 0
    while duration_seconds is None or loop_second < duration_seconds:
        observed_at = start_at + timedelta(seconds=loop_second)
        for assignment in assignments:
            dataset = datasets[assignment.scenario_id]
            source = dataset[
                (loop_second + assignment.offset_seconds) % len(dataset)
            ]
            frame = source.model_copy(
                update={
                    "vehicle_id": assignment.vehicle_id,
                    "observed_at": observed_at,
                    "sequence": loop_second,
                }
            )
            await store.publish_live_only(frame)
        loop_second += 1
        if duration_seconds is not None and loop_second >= duration_seconds:
            break
        await asyncio.sleep(1.0 / speed)
    return loop_second


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay pre-generated 1-hour scenario datasets for all vehicles"
    )
    parser.add_argument(
        "--scenario-dir",
        default=str(Path.cwd() / "runtime" / "scenarios"),
        help="directory containing scenario frames.jsonl datasets",
    )
    parser.add_argument(
        "--assignments-file",
        default=None,
        help="JSON assignment file; when omitted, assignments are read from RDS",
    )
    parser.add_argument("--start-at", type=_parse_aware, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="logical seconds to replay (default: run forever)",
    )
    return parser.parse_args()


async def _run_replay(args: argparse.Namespace) -> None:
    settings = Settings.load()
    datasets = load_datasets(Path(args.scenario_dir))
    if args.assignments_file:
        assignments = load_assignments_from_file(Path(args.assignments_file))
    else:
        engine, sessions = create_database(settings.database_url)
        try:
            assignments = await load_vehicle_assignments_from_db(sessions)
        finally:
            await engine.dispose()
    if not assignments:
        raise SystemExit("no vehicle assignments to replay")

    redis = Redis.from_url(settings.redis_url, decode_responses=False)
    store = TwinRedisStore(redis)
    try:
        await store.ping()
        start_at = args.start_at or datetime.now(timezone.utc).replace(
            microsecond=0
        )
        count = await replay_scenarios(
            store,
            assignments,
            datasets,
            start_at=start_at,
            speed=args.speed,
            duration_seconds=args.duration,
        )
        print(
            f"replayed {count} logical seconds for {len(assignments)} vehicles"
        )
    finally:
        await redis.aclose()


def main() -> None:
    asyncio.run(_run_replay(parse_args()))


if __name__ == "__main__":
    main()
