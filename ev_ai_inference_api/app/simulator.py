from __future__ import annotations

import argparse
import asyncio
import math
import os
from dataclasses import dataclass
from datetime import datetime, time as wall_clock_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from pydantic import TypeAdapter
from redis.asyncio import Redis
from sqlalchemy import delete

from app.core.config import Settings
from app.core.twin_redis import RISK_SORTED_SET, TwinRedisStore, latest_key, prebuffer_key
from app.db.models import (
    INCIDENT_TYPE_BATTERY,
    INCIDENT_TYPE_CONNECTOR,
    TwinIncident,
)
from app.db.repository import TwinRepository
from app.db.session import create_database
from app.demo_profiles import DemoProfileFrame, load_demo_profiles, profile_frame_at
from app.schemas.twins import TwinFrame, TwinSampleRequest
from app.services.twin_service import temperature_level, voltage_level
from app.services.thermal_inference import ThermalInferenceClient
from app.services.thermal_render import (
    module_scores_from_cells,
    render_thermal_frame,
    sensor_cell_heat_scores,
)
from app.services.twin_fusion import fuse_twin_state
from app.workers.twin_persistence import deterministic_incident_id


KOREA_TIMEZONE = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True)
class VehicleDemoProfile:
    vehicle_id: str
    risk_level: int
    abnormal_type: str
    incident_type: str | None
    charging_time: wall_clock_time


VEHICLE_PROFILES = (
    VehicleDemoProfile(
        "car-uuid-001", 3, "temperature_rise", INCIDENT_TYPE_BATTERY,
        wall_clock_time(14, 0, 0),
    ),
    VehicleDemoProfile(
        "car-uuid-002", 2, "fire_risk", INCIDENT_TYPE_CONNECTOR,
        wall_clock_time(14, 0, 0),
    ),
    VehicleDemoProfile(
        "car-uuid-003", 1, "temperature_rise", INCIDENT_TYPE_BATTERY,
        wall_clock_time(14, 0, 0),
    ),
    VehicleDemoProfile(
        "car-uuid-004", 0, "normal", None,
        wall_clock_time(14, 12, 30),
    ),
    VehicleDemoProfile(
        "car-uuid-005", 2, "overcharge_warning", INCIDENT_TYPE_BATTERY,
        wall_clock_time(14, 25, 10),
    ),
    VehicleDemoProfile(
        "car-uuid-006", 0, "normal", None,
        wall_clock_time(14, 30, 0),
    ),
    VehicleDemoProfile(
        "car-uuid-007", 3, "fire_risk", INCIDENT_TYPE_CONNECTOR,
        wall_clock_time(14, 40, 15),
    ),
    VehicleDemoProfile(
        "car-uuid-008", 0, "normal", None,
        wall_clock_time(14, 45, 0),
    ),
    VehicleDemoProfile(
        "car-uuid-009", 1, "temperature_rise", INCIDENT_TYPE_BATTERY,
        wall_clock_time(15, 0, 0),
    ),
    VehicleDemoProfile(
        "car-uuid-010", 0, "normal", None,
        wall_clock_time(15, 10, 20),
    ),
)
VEHICLE_IDS = tuple(profile.vehicle_id for profile in VEHICLE_PROFILES)
VEHICLE_RISK_LEVELS = tuple(profile.risk_level for profile in VEHICLE_PROFILES)
SEED_FRAME_COUNT = 10_800
LIVE_FRAME_COUNT = 10_801
SEED_TRIGGER_FRAME_INDEX = 3_599
_AWARE_DATETIME = TypeAdapter(datetime)


def _parse_aware(value: str) -> datetime:
    parsed = _AWARE_DATETIME.validate_python(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone offset")
    return parsed


def _stage_temperature_decic(level: int) -> int:
    return (350, 500, 650, 850)[level]


def _incident_trigger_at(
    start_at: datetime,
    profile: VehicleDemoProfile,
) -> datetime:
    local_date = start_at.astimezone(KOREA_TIMEZONE).date()
    local_trigger = datetime.combine(
        local_date,
        profile.charging_time,
        tzinfo=KOREA_TIMEZONE,
    )
    return local_trigger.astimezone(timezone.utc)


def _cap_profile_frame(
    frame: DemoProfileFrame,
    max_risk_level: int,
) -> DemoProfileFrame:
    """Keep the original spatial gradient while matching the car-list stage."""
    if not 0 <= max_risk_level <= 3:
        raise ValueError("max_risk_level must be between 0 and 3")

    temperature_limit = (449, 599, 799, None)[max_risk_level]
    voltage_floor = (2_400, 2_001, 1_501, 0)[max_risk_level]
    voltage_ceiling = (4_350, 4_399, 4_499, 5_000)[max_risk_level]

    temperatures = (
        None
        if frame.temperature_decic is None
        else tuple(
            min(value, temperature_limit)
            if temperature_limit is not None
            else value
            for value in frame.temperature_decic
        )
    )
    voltages = (
        None
        if frame.voltage_mv is None
        else tuple(
            min(voltage_ceiling, max(voltage_floor, value))
            for value in frame.voltage_mv
        )
    )
    connector_temperatures = (
        None
        if frame.connector_temperature_decic is None
        else tuple(
            min(value, temperature_limit)
            if temperature_limit is not None
            else value
            for value in frame.connector_temperature_decic
        )
    )
    state_level = (
        None
        if frame.state_level is None
        else tuple(min(value, max_risk_level) for value in frame.state_level)
    )
    connector_state_level = (
        None
        if frame.connector_state_level is None
        else tuple(
            min(value, max_risk_level)
            for value in frame.connector_state_level
        )
    )
    return DemoProfileFrame(
        risk_level=min(frame.risk_level, max_risk_level),
        temperature_decic=temperatures,
        voltage_mv=voltages,
        state_level=state_level,
        hotspot_cell_index=frame.hotspot_cell_index,
        connector_temperature_decic=connector_temperatures,
        connector_state_level=connector_state_level,
        hotspot_connector_index=frame.hotspot_connector_index,
    )


def _clamp_frame_risk(frame: TwinFrame, max_risk_level: int) -> TwinFrame:
    state_level = [
        min(value, max_risk_level) for value in frame.state_level
    ]
    connector_state_level = [
        min(value, max_risk_level)
        for value in frame.connector_state_level
    ]
    module_state_level = [
        max(state_level[index : index + 8])
        for index in range(0, 96, 8)
    ]
    return frame.model_copy(
        update={
            "state_level": state_level,
            "connector_state_level": connector_state_level,
            "ml_risk_level": min(
                frame.ml_risk_level or 0,
                max_risk_level,
            ),
            "physics_risk_level": min(
                frame.physics_risk_level,
                max_risk_level,
            ),
            "final_risk_level": min(
                frame.final_risk_level,
                max_risk_level,
            ),
            "image_risk_level": (
                None
                if frame.image_risk_level is None
                else min(frame.image_risk_level, max_risk_level)
            ),
            "module_state_level": module_state_level,
            "hotspot_module_index": max(
                range(12),
                key=module_state_level.__getitem__,
            ),
        }
    )


def sample_arrays(
    *,
    vehicle_number: int,
    sequence: int,
    risk_level: int,
    incident_type: str = "combined",
) -> tuple[list[int], list[int], list[int]]:
    if incident_type not in {"combined", INCIDENT_TYPE_CONNECTOR, INCIDENT_TYPE_BATTERY}:
        raise ValueError(f"unsupported incident_type: {incident_type}")
    temperatures = [
        320 + round(5 * math.sin((sequence + index * 7 + vehicle_number) / 41.0))
        for index in range(96)
    ]
    voltages = [
        3_820 + round(10 * math.sin((sequence + index * 11) / 53.0))
        for index in range(96)
    ]
    hotspot = 51 + vehicle_number
    battery_risk_level = (
        risk_level if incident_type in {"combined", INCIDENT_TYPE_BATTERY} else 0
    )
    connector_risk_level = (
        risk_level if incident_type in {"combined", INCIDENT_TYPE_CONNECTOR} else 0
    )
    battery_indexes = [hotspot]
    if incident_type == INCIDENT_TYPE_BATTERY:
        module_start = (hotspot // 8) * 8
        battery_indexes = list(range(module_start, module_start + 8))
    risk_voltage = (3_820, 2_300, 1_900, 1_400)[battery_risk_level]
    for index in battery_indexes:
        temperatures[index] = _stage_temperature_decic(battery_risk_level)
        voltages[index] = risk_voltage
    connector_peak = _stage_temperature_decic(connector_risk_level)
    if incident_type == INCIDENT_TYPE_CONNECTOR and connector_risk_level > 0:
        connector = [
            connector_peak,
            max(300, connector_peak - 20),
            max(300, connector_peak - 40),
        ]
    else:
        connector = [connector_peak, max(300, connector_peak - 120), 300]
    return temperatures, voltages, connector


def build_sample(
    *,
    vehicle_number: int,
    observed_at: datetime,
    sequence: int,
    risk_level: int,
    incident_type: str = "combined",
) -> TwinSampleRequest:
    temperatures, voltages, connector = sample_arrays(
        vehicle_number=vehicle_number,
        sequence=sequence,
        risk_level=risk_level,
        incident_type=incident_type,
    )
    return TwinSampleRequest(
        observed_at=observed_at,
        sequence=sequence,
        temperature_decic=temperatures,
        voltage_mv=voltages,
        connector_temperature_decic=connector,
        ambient_temperature_c=25.0,
        pack_current_a=90.0,
    )


def build_seed_frame(
    *,
    vehicle_id: str,
    vehicle_number: int,
    observed_at: datetime,
    sequence: int,
    risk_level: int,
    incident_type: str,
    profile_frame: DemoProfileFrame | None = None,
) -> TwinFrame:
    if (
        incident_type in {INCIDENT_TYPE_CONNECTOR, INCIDENT_TYPE_BATTERY}
        and profile_frame is None
    ):
        raise ValueError("typed seed frames require an original demo profile frame")
    sample = build_sample(
        vehicle_number=vehicle_number,
        observed_at=observed_at,
        sequence=sequence,
        risk_level=(0 if profile_frame is not None else risk_level),
        incident_type=incident_type,
    )
    temperatures = list(sample.temperature_decic)
    voltages = list(sample.voltage_mv)
    connector_temperatures = list(sample.connector_temperature_decic)
    cell_levels = [
        max(temperature_level(temperature), voltage_level(voltage))
        for temperature, voltage in zip(
            temperatures, voltages, strict=True
        )
    ]
    connector_levels = [
        temperature_level(value) for value in connector_temperatures
    ]
    hotspot_cell_index = max(range(96), key=temperatures.__getitem__)
    hotspot_connector_index = max(
        range(3), key=connector_temperatures.__getitem__
    )

    if profile_frame is not None and incident_type == INCIDENT_TYPE_BATTERY:
        if (
            profile_frame.temperature_decic is None
            or profile_frame.voltage_mv is None
            or profile_frame.state_level is None
            or profile_frame.hotspot_cell_index is None
        ):
            raise ValueError("battery demo profile frame is incomplete")
        temperatures = list(profile_frame.temperature_decic)
        voltages = list(profile_frame.voltage_mv)
        cell_levels = list(profile_frame.state_level)
        hotspot_cell_index = profile_frame.hotspot_cell_index
    elif profile_frame is not None and incident_type == INCIDENT_TYPE_CONNECTOR:
        if (
            profile_frame.connector_temperature_decic is None
            or profile_frame.connector_state_level is None
            or profile_frame.hotspot_connector_index is None
        ):
            raise ValueError("connector demo profile frame is incomplete")
        connector_temperatures = list(profile_frame.connector_temperature_decic)
        connector_levels = list(profile_frame.connector_state_level)
        hotspot_connector_index = profile_frame.hotspot_connector_index

    local_risk_level = max(
        profile_frame.risk_level if profile_frame is not None else risk_level,
        max(cell_levels),
        max(connector_levels),
    )
    cell_heat_score = list(
        sensor_cell_heat_scores(temperatures, ambient=25.0)
    )
    module_heat_score = list(module_scores_from_cells(cell_heat_score))
    module_state_level = [
        max(cell_levels[index : index + 8]) for index in range(0, 96, 8)
    ]
    hotspot_cell_index = max(
        range(96), key=cell_heat_score.__getitem__
    )
    return TwinFrame(
        vehicle_id=vehicle_id,
        observed_at=observed_at,
        sequence=sequence,
        temperature_decic=temperatures,
        voltage_mv=voltages,
        state_level=cell_levels,
        connector_temperature_decic=connector_temperatures,
        connector_state_level=connector_levels,
        hotspot_cell_index=hotspot_cell_index,
        hotspot_connector_index=hotspot_connector_index,
        ml_risk_level=(
            0 if incident_type == INCIDENT_TYPE_CONNECTOR else max(cell_levels)
        ),
        physics_risk_level=max(max(cell_levels), max(connector_levels)),
        final_risk_level=local_risk_level,
        cell_heat_score=cell_heat_score,
        module_heat_score=module_heat_score,
        module_state_level=module_state_level,
        hotspot_module_index=hotspot_cell_index // 8,
    )


def _sample_from_frame(frame: TwinFrame) -> TwinSampleRequest:
    return TwinSampleRequest(
        observed_at=frame.observed_at,
        sequence=frame.sequence,
        temperature_decic=list(frame.temperature_decic),
        voltage_mv=list(frame.voltage_mv),
        connector_temperature_decic=list(frame.connector_temperature_decic),
        ambient_temperature_c=25.0,
        pack_current_a=90.0,
    )


async def _decorate_seed_frame(
    frame: TwinFrame,
    *,
    incident_id: str,
    incident_type: str,
    max_risk_level: int,
    inference: ThermalInferenceClient,
    thermal_root: Path,
) -> TwinFrame:
    if incident_type != INCIDENT_TYPE_BATTERY or frame.sequence % 30 != 29:
        return frame
    sample = _sample_from_frame(frame)
    rendered = render_thermal_frame(frame.vehicle_id, sample)
    relative_ref = Path("thermal") / incident_id / f"{frame.sequence:05d}.png"
    target = thermal_root / relative_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(rendered.image_bytes)
    result = await inference.infer(
        vehicle_id=frame.vehicle_id,
        observed_at=frame.observed_at.isoformat(),
        sequence=frame.sequence,
        layout_id=frame.layout_id,
        image_bytes=rendered.image_bytes,
    )
    fused = fuse_twin_state(
        sample,
        ml_level=frame.ml_risk_level,
        physical_rule_level=frame.physics_risk_level,
        bms_final_level=frame.final_risk_level,
        thermal_result=result,
        thermal_cell_heat_score=rendered.cell_heat_score,
        thermal_frame_ref=relative_ref.as_posix(),
        thermal_frame_sha256=rendered.sha256,
    )
    return _clamp_frame_risk(frame.model_copy(
        update={
            "state_level": fused["state_level"],
            "connector_state_level": fused["connector_state_level"],
            "physics_risk_level": fused["physics_risk_level"],
            "final_risk_level": fused["final_risk_level"],
            "image_risk_level": fused["image_risk_level"],
            "image_confidence": fused["image_confidence"],
            "image_probabilities": fused["image_probabilities"],
            "image_model_status": fused["image_model_status"],
            "cell_heat_score": fused["cell_heat_score"],
            "module_heat_score": fused["module_heat_score"],
            "module_state_level": fused["module_state_level"],
            "hotspot_module_index": fused["hotspot_module_index"],
            "thermal_frame_ref": fused["thermal_frame_ref"],
            "thermal_frame_sha256": fused["thermal_frame_sha256"],
            "fusion_source": fused["fusion_source"],
        }
    ), max_risk_level)


async def seed_history(start_at: datetime) -> None:
    settings = Settings.load()
    engine, sessions = create_database(settings.database_url)
    redis = Redis.from_url(settings.redis_url, decode_responses=False)
    store = TwinRedisStore(redis)
    repository = TwinRepository()
    demo_profiles = load_demo_profiles()
    inference = ThermalInferenceClient(
        settings.thermal_inference_url,
        settings.thermal_inference_token,
        settings.thermal_inference_timeout_seconds,
    )
    thermal_root = Path(
        os.getenv("THERMAL_DATA_DIR", str(Path.cwd() / "runtime" / "thermal"))
    )
    try:
        await store.ping()
        async with sessions() as session:
            async with session.begin():
                await session.execute(
                    delete(TwinIncident).where(
                        TwinIncident.vehicle_id.in_(VEHICLE_IDS)
                    )
                )
        for vehicle_number, profile in enumerate(VEHICLE_PROFILES):
            if profile.incident_type is None:
                continue
            incident_type = profile.incident_type
            demo_profile = demo_profiles[incident_type]
            triggered_at = _incident_trigger_at(start_at, profile)
            incident_start = triggered_at - timedelta(hours=1)
            incident_id = deterministic_incident_id(profile.vehicle_id, triggered_at)
            frames: list[TwinFrame] = []
            for index in range(SEED_FRAME_COUNT):
                source_frame = profile_frame_at(
                    demo_profile,
                    index,
                    output_frame_count=SEED_FRAME_COUNT,
                    trigger_frame_index=SEED_TRIGGER_FRAME_INDEX,
                )
                profile_frame = _cap_profile_frame(
                    source_frame,
                    profile.risk_level,
                )
                frame = build_seed_frame(
                    vehicle_id=profile.vehicle_id,
                    vehicle_number=vehicle_number,
                    observed_at=incident_start + timedelta(seconds=index),
                    sequence=index,
                    risk_level=profile.risk_level,
                    incident_type=incident_type,
                    profile_frame=profile_frame,
                )
                frame = await _decorate_seed_frame(
                    frame,
                    incident_id=incident_id,
                    incident_type=incident_type,
                    max_risk_level=profile.risk_level,
                    inference=inference,
                    thermal_root=thermal_root,
                )
                frames.append(frame)
            async with sessions() as session:
                async with session.begin():
                    await repository.create_incident(
                        session,
                        incident_id=incident_id,
                        vehicle_id=profile.vehicle_id,
                        triggered_at=triggered_at,
                        window_start=incident_start,
                        window_end=incident_start + timedelta(
                            seconds=SEED_FRAME_COUNT
                        ),
                        incident_type=incident_type,
                    )
                    await repository.replace_seed_frames(session, incident_id, frames)
                    await repository.mark_complete(
                        session,
                        incident_id,
                        incident_start + timedelta(seconds=SEED_FRAME_COUNT),
                    )
            print(
                f"seeded {profile.vehicle_id}: type={incident_type} "
                f"incident={incident_id} frames={SEED_FRAME_COUNT}"
            )

        latest_at = datetime.now(timezone.utc).replace(microsecond=0)
        for vehicle_number, profile in enumerate(VEHICLE_PROFILES):
            if profile.incident_type is None:
                latest_frame = build_seed_frame(
                    vehicle_id=profile.vehicle_id,
                    vehicle_number=vehicle_number,
                    observed_at=latest_at,
                    sequence=SEED_FRAME_COUNT - 1,
                    risk_level=0,
                    incident_type="combined",
                )
            else:
                demo_profile = demo_profiles[profile.incident_type]
                source_frame = _cap_profile_frame(
                    demo_profile.frames[-1],
                    profile.risk_level,
                )
                latest_frame = build_seed_frame(
                    vehicle_id=profile.vehicle_id,
                    vehicle_number=vehicle_number,
                    observed_at=latest_at,
                    sequence=SEED_FRAME_COUNT - 1,
                    risk_level=profile.risk_level,
                    incident_type=profile.incident_type,
                    profile_frame=source_frame,
                )
            await store.seed_latest(latest_frame)
    finally:
        await redis.aclose()
        await engine.dispose()


async def reset_live_state(client: httpx.AsyncClient) -> None:
    settings = Settings.load()
    redis = Redis.from_url(settings.redis_url, decode_responses=False)
    try:
        pipe = redis.pipeline(transaction=True)
        for vehicle_id in VEHICLE_IDS:
            pipe.delete(latest_key(vehicle_id), prebuffer_key(vehicle_id))
            pipe.zrem(RISK_SORTED_SET, vehicle_id)
        await pipe.execute()
        for vehicle_id in VEHICLE_IDS:
            response = await client.post(f"/v1/vehicles/{vehicle_id}/reset")
            response.raise_for_status()
    finally:
        await redis.aclose()


async def replay_live(
    *,
    api_url: str,
    start_at: datetime,
    speed: float,
    keep_state: bool,
    with_thermal: bool = False,
) -> None:
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be a positive finite number")
    demo_profiles = load_demo_profiles()
    async with httpx.AsyncClient(base_url=api_url, timeout=60.0) as client:
        if not keep_state:
            await reset_live_state(client)
        for index in range(LIVE_FRAME_COUNT):
            risk_active = index >= 3_600
            observed_at = start_at + timedelta(seconds=index)
            payloads: list[TwinSampleRequest] = []
            for vehicle_number, profile in enumerate(VEHICLE_PROFILES):
                if not risk_active or profile.incident_type is None:
                    payloads.append(
                        build_sample(
                            vehicle_number=vehicle_number,
                            observed_at=observed_at,
                            sequence=index,
                            risk_level=0,
                            incident_type="combined",
                        )
                    )
                    continue
                demo_profile = demo_profiles[profile.incident_type]
                source_frame = profile_frame_at(
                    demo_profile,
                    index,
                    output_frame_count=LIVE_FRAME_COUNT,
                    trigger_frame_index=SEED_TRIGGER_FRAME_INDEX,
                )
                profile_frame = _cap_profile_frame(
                    source_frame,
                    profile.risk_level,
                )
                twin_frame = build_seed_frame(
                    vehicle_id=profile.vehicle_id,
                    vehicle_number=vehicle_number,
                    observed_at=observed_at,
                    sequence=index,
                    risk_level=profile.risk_level,
                    incident_type=profile.incident_type,
                    profile_frame=profile_frame,
                )
                payloads.append(_sample_from_frame(twin_frame))
            requests = []
            for vehicle_id, payload in zip(VEHICLE_IDS, payloads, strict=True):
                if with_thermal:
                    rendered = render_thermal_frame(vehicle_id, payload)
                    requests.append(
                        client.post(
                            f"/api/v1/twins/vehicles/{vehicle_id}/observations",
                            data={"sample_json": payload.model_dump_json()},
                            files={
                                "thermal_image": (
                                    f"{vehicle_id}-{index}.png",
                                    rendered.image_bytes,
                                    "image/png",
                                )
                            },
                        )
                    )
                else:
                    requests.append(
                        client.post(
                            f"/api/v1/twins/vehicles/{vehicle_id}/samples",
                            json=payload.model_dump(mode="json"),
                        )
                    )
            responses = await asyncio.gather(*requests)
            for response in responses:
                response.raise_for_status()
            if index % 300 == 0 or index == LIVE_FRAME_COUNT - 1:
                print(f"replayed logical second {index}/{LIVE_FRAME_COUNT - 1}")
            if index < LIVE_FRAME_COUNT - 1:
                await asyncio.sleep(1.0 / speed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local 3D twin demonstration data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser(
        "seed-history",
        description=(
            "Insert profile-matched 10,800-frame incidents for six risk vehicles"
        ),
    )
    seed.add_argument(
        "--start-at",
        type=_parse_aware,
        default=datetime(2026, 7, 31, tzinfo=timezone.utc),
    )

    live = subparsers.add_parser(
        "replay-live", description="POST ten profile-matched logical 1 Hz streams"
    )
    live.add_argument("--api-url", default="http://127.0.0.1:8000")
    live.add_argument(
        "--start-at",
        type=_parse_aware,
        default=None,
    )
    live.add_argument("--speed", type=float, default=60.0)
    live.add_argument(
        "--keep-state",
        action="store_true",
        help="Do not clear vehicle latest/prebuffer and model sessions first",
    )
    live.add_argument(
        "--with-thermal",
        action="store_true",
        help="Render a synchronized thermal PNG and use the observation endpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "seed-history":
        asyncio.run(seed_history(args.start_at))
        return
    start_at = args.start_at or datetime.now(timezone.utc).replace(microsecond=0)
    asyncio.run(
        replay_live(
            api_url=args.api_url,
            start_at=start_at,
            speed=args.speed,
            keep_state=args.keep_state,
            with_thermal=args.with_thermal,
        )
    )


if __name__ == "__main__":
    main()
