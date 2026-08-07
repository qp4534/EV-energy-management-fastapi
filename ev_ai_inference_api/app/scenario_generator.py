from __future__ import annotations

import argparse
import asyncio
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from app.core.config import Settings, validate_bundle
from app.core.session_manager import SessionManager
from app.scenario_catalog import (
    SCENARIO_BY_ID,
    SCENARIOS,
    ScenarioDefinition,
)
from app.simulator import _clamp_frame_risk
from app.schemas.twins import (
    CELL_COUNT,
    CONNECTOR_COMPONENT_COUNT,
    TwinFrame,
    TwinSampleRequest,
)
from app.services.current_stage_service import CurrentStageService
from app.services.thermal_inference import ThermalInferenceClient
from app.services.thermal_render import render_thermal_frame
from app.services.twin_fusion import fuse_twin_state
from app.services.twin_service import _STAGE_LEVEL, build_model_request


SCENARIO_FRAME_COUNT = 3_600  # 1 hour at 1 Hz
HISTORY_FRAME_COUNT = 10_800  # incident -1h ~ +2h at 1 Hz
THERMAL_INTERVAL_SECONDS = 5
HOTSPOT_CELL_INDEX = 51
HOTSPOT_MODULE_START = (HOTSPOT_CELL_INDEX // 8) * 8

_AWARE_DATETIME = TypeAdapter(datetime)


def _parse_aware(value: str) -> datetime:
    parsed = _AWARE_DATETIME.validate_python(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone offset")
    return parsed


def _ramp(progress: float, exponent: float = 1.5) -> float:
    clamped = min(1.0, max(0.0, progress))
    return clamped**exponent


def _base_temperatures(sequence: int, vehicle_number: int) -> list[int]:
    return [
        310 + round(5 * math.sin((sequence + index * 7 + vehicle_number) / 41.0))
        for index in range(CELL_COUNT)
    ]


def _base_voltages(sequence: int, vehicle_number: int) -> list[int]:
    return [
        3_810 + round(8 * math.sin((sequence + index * 11) / 53.0))
        for index in range(CELL_COUNT)
    ]


def _base_connector(sequence: int) -> list[int]:
    return [
        300 + round(2 * math.sin((sequence + component * 13) / 37.0))
        for component in range(CONNECTOR_COMPONENT_COUNT)
    ]


def build_scenario_sample(
    scenario: ScenarioDefinition,
    sequence: int,
    *,
    frame_count: int = SCENARIO_FRAME_COUNT,
    vehicle_number: int = 0,
) -> TwinSampleRequest:
    """Build one 1 Hz sensor sample for a scenario at a given sequence index."""

    if not 0 <= sequence < frame_count:
        raise ValueError("sequence must be inside the scenario frame window")
    progress = sequence / max(1, frame_count - 1)
    temperatures = _base_temperatures(sequence, vehicle_number)
    voltages = _base_voltages(sequence, vehicle_number)
    connector = _base_connector(sequence)
    pack_current = 90.0

    scenario_id = scenario.scenario_id
    if scenario_id == "connector_local_overheat":
        intensity = _ramp(progress)
        connector = [
            300 + round(568 * intensity),
            300 + round(431 * intensity),
            300 + round(181 * intensity),
        ]
    elif scenario_id == "battery_over_temp":
        intensity = _ramp(progress)
        for index in range(HOTSPOT_MODULE_START, HOTSPOT_MODULE_START + 8):
            temperatures[index] = 310 + round(540 * intensity)
            voltages[index] = 3_810 - round(2_400 * intensity)
    elif scenario_id == "thermal_runaway_risk":
        intensity = _ramp(progress, exponent=1.2)
        for index in range(40, 72):
            temperatures[index] = 310 + round(590 * intensity)
            voltages[index] = 3_810 - round(2_500 * intensity)
    elif scenario_id == "cell_voltage_imbalance":
        intensity = _ramp(progress)
        voltages[HOTSPOT_CELL_INDEX] = 3_810 - round(2_000 * intensity)
    elif scenario_id == "battery_overheat_sign":
        intensity = _ramp(progress)
        for index in range(HOTSPOT_MODULE_START, HOTSPOT_MODULE_START + 8):
            temperatures[index] = 310 + round(340 * intensity)
    elif scenario_id == "rapid_temp_rise":
        late_progress = _ramp(max(0.0, (progress - 0.70) / 0.30))
        for index in range(HOTSPOT_MODULE_START, HOTSPOT_MODULE_START + 8):
            temperatures[index] = 310 + round(340 * late_progress)
    elif scenario_id == "connector_temp_rise":
        intensity = _ramp(progress)
        connector = [
            300 + round(250 * intensity),
            300 + round(150 * intensity),
            300 + round(80 * intensity),
        ]
    elif scenario_id == "cell_voltage_deviation":
        intensity = _ramp(progress)
        voltages[HOTSPOT_CELL_INDEX] = 3_810 - round(1_500 * intensity)
    elif scenario_id == "charging_current_fluctuation":
        pack_current = (
            90.0
            + 60.0 * math.sin(sequence / 5.0)
            + 25.0 * math.sin(sequence / 17.0)
        )

    return TwinSampleRequest(
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        sequence=sequence,
        temperature_decic=temperatures,
        voltage_mv=voltages,
        connector_temperature_decic=connector,
        ambient_temperature_c=25.0,
        pack_current_a=pack_current,
    )


def _supervisor_factory(settings: Settings):
    from hybrid_safety_supervisor import HybridSafetySupervisorV2

    return lambda: HybridSafetySupervisorV2(
        settings.bundle_dir / "models" / "hybrid_v1",
        settings.bundle_dir / "config" / "safety_policy.v2.json",
    )


async def generate_scenario_frames(
    scenario: ScenarioDefinition,
    *,
    frame_count: int = SCENARIO_FRAME_COUNT,
    start_at: datetime | None = None,
    vehicle_id: str | None = None,
    settings: Settings | None = None,
    thermal_client: ThermalInferenceClient | None = None,
    thermal_root: Path | None = None,
    render_thermal: bool = False,
) -> list[TwinFrame]:
    """Run one scenario through the BMS supervisor and return final TwinFrames."""

    settings = settings or Settings.load()
    validate_bundle(settings)
    resolved_start = start_at or datetime.now(timezone.utc).replace(microsecond=0)
    resolved_vehicle = vehicle_id or f"scenario-{scenario.scenario_id}"
    sessions = SessionManager(
        _supervisor_factory(settings),
        settings.session_ttl_seconds,
        settings.max_sessions,
    )
    current_stage = CurrentStageService(sessions)
    client = thermal_client or ThermalInferenceClient(
        settings.thermal_inference_url,
        settings.thermal_inference_token,
        settings.thermal_inference_timeout_seconds,
    )

    frames: list[TwinFrame] = []
    for index in range(frame_count):
        payload = build_scenario_sample(
            scenario,
            index,
            frame_count=frame_count,
        ).model_copy(
            update={
                "observed_at": resolved_start + timedelta(seconds=index),
                "sequence": index,
            }
        )
        model_request = build_model_request(payload)
        result = await current_stage.evaluate(resolved_vehicle, model_request)
        ml_level = (
            None
            if result.ml_pattern_stage is None
            else _STAGE_LEVEL[result.ml_pattern_stage]
        )
        physics_level = _STAGE_LEVEL[result.physical_rule_level]
        bms_final_level = _STAGE_LEVEL[result.final_safety_alert]

        thermal_result = None
        thermal_cell_heat_score = None
        thermal_frame_ref = None
        thermal_frame_sha256 = None
        if render_thermal and index % THERMAL_INTERVAL_SECONDS == 0:
            rendered = render_thermal_frame(resolved_vehicle, payload)
            thermal_cell_heat_score = rendered.cell_heat_score
            thermal_frame_sha256 = rendered.sha256
            thermal_frame_ref = (
                f"thermal/{scenario.scenario_id}/{index:05d}.png"
            )
            if thermal_root is not None:
                target = thermal_root / thermal_frame_ref
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(rendered.image_bytes)
            if client.enabled:
                thermal_result = await client.infer(
                    vehicle_id=resolved_vehicle,
                    observed_at=payload.observed_at.isoformat(),
                    sequence=index,
                    layout_id=payload.layout_id,
                    image_bytes=rendered.image_bytes,
                )

        fused = fuse_twin_state(
            payload,
            ml_level=ml_level,
            physical_rule_level=physics_level,
            bms_final_level=bms_final_level,
            thermal_result=thermal_result,
            thermal_cell_heat_score=thermal_cell_heat_score,
            thermal_frame_ref=thermal_frame_ref,
            thermal_frame_sha256=thermal_frame_sha256,
        )
        final_level = max(bms_final_level, scenario.risk_level)
        frame = TwinFrame(
            vehicle_id=resolved_vehicle,
            observed_at=payload.observed_at,
            sequence=index,
            temperature_decic=list(payload.temperature_decic),
            voltage_mv=list(payload.voltage_mv),
            state_level=fused["state_level"],
            connector_temperature_decic=list(
                payload.connector_temperature_decic
            ),
            connector_state_level=fused["connector_state_level"],
            hotspot_cell_index=fused["hotspot_cell_index"],
            hotspot_connector_index=max(
                range(len(payload.connector_temperature_decic)),
                key=payload.connector_temperature_decic.__getitem__,
            ),
            ml_risk_level=ml_level,
            physics_risk_level=physics_level,
            final_risk_level=final_level,
            cell_heat_score=fused["cell_heat_score"],
            image_risk_level=fused["image_risk_level"],
            image_confidence=fused["image_confidence"],
            image_probabilities=fused["image_probabilities"],
            image_model_status=fused["image_model_status"],
            module_heat_score=fused["module_heat_score"],
            module_state_level=fused["module_state_level"],
            hotspot_module_index=fused["hotspot_module_index"],
            thermal_frame_ref=thermal_frame_ref,
            thermal_frame_sha256=thermal_frame_sha256,
            fusion_source=fused["fusion_source"],
        )
        frame = _clamp_frame_risk(frame, scenario.risk_level)
        frames.append(frame)
    return frames


def write_scenario_dataset(
    scenario: ScenarioDefinition,
    frames: list[TwinFrame],
    out_dir: Path,
    *,
    metadata_extra: dict[str, Any] | None = None,
) -> Path:
    """Write frames.jsonl + metadata.json for one scenario dataset."""

    scenario_dir = Path(out_dir) / scenario.scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)
    frames_path = scenario_dir / "frames.jsonl"
    with frames_path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(frame.model_dump_json() + "\n")
    metadata = {
        "schema_version": 1,
        "scenario_id": scenario.scenario_id,
        "name": scenario.name,
        "abnormal_type": scenario.abnormal_type,
        "risk_level": scenario.risk_level,
        "incident_type": scenario.incident_type,
        "source_type": scenario.source_type,
        "description": scenario.description,
        "frame_count": len(frames),
        "frame_interval_seconds": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **(metadata_extra or {}),
    }
    (scenario_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return scenario_dir


def load_scenario_frames(path: Path) -> list[TwinFrame]:
    """Load a previously generated frames.jsonl dataset."""

    frames: list[TwinFrame] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                frames.append(TwinFrame.model_validate_json(stripped))
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pre-computed 1-hour digital-twin scenario datasets"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="print the scenario catalog")

    generate = subparsers.add_parser(
        "generate",
        help="generate scenario datasets through the BMS supervisor",
    )
    generate.add_argument(
        "--scenario",
        default="all",
        help="scenario id or 'all' (default: all)",
    )
    generate.add_argument(
        "--frame-count",
        type=int,
        default=SCENARIO_FRAME_COUNT,
        help="frames per scenario (default: 3600 = 1 hour)",
    )
    generate.add_argument("--start-at", type=_parse_aware, default=None)
    generate.add_argument(
        "--out-dir",
        default=str(Path.cwd() / "runtime" / "scenarios"),
        help="output directory (default: runtime/scenarios)",
    )
    generate.add_argument(
        "--with-thermal",
        action="store_true",
        help="render thermal frames and call the thermal worker when configured",
    )
    generate.add_argument(
        "--thermal-root",
        default=None,
        help="directory for rendered thermal frames (default: out-dir/thermal)",
    )
    return parser.parse_args()


async def _run_generate(args: argparse.Namespace) -> None:
    settings = Settings.load()
    validate_bundle(settings)
    if args.scenario == "all":
        scenarios = SCENARIOS
    elif args.scenario in SCENARIO_BY_ID:
        scenarios = (SCENARIO_BY_ID[args.scenario],)
    else:
        raise SystemExit(f"unknown scenario id: {args.scenario}")

    start_at = args.start_at or datetime.now(timezone.utc).replace(microsecond=0)
    out_dir = Path(args.out_dir)
    thermal_root = (
        Path(args.thermal_root)
        if args.thermal_root
        else out_dir / "thermal"
    )
    thermal_client = ThermalInferenceClient(
        settings.thermal_inference_url,
        settings.thermal_inference_token,
        settings.thermal_inference_timeout_seconds,
    )
    for scenario in scenarios:
        frames = await generate_scenario_frames(
            scenario,
            frame_count=args.frame_count,
            start_at=start_at,
            settings=settings,
            thermal_client=thermal_client,
            thermal_root=thermal_root,
            render_thermal=args.with_thermal,
        )
        scenario_dir = write_scenario_dataset(
            scenario,
            frames,
            out_dir,
            metadata_extra={
                "start_at": start_at.isoformat(),
                "model_bundle": str(settings.bundle_dir),
                "with_thermal": args.with_thermal,
            },
        )
        final_level = frames[-1].final_risk_level if frames else None
        print(
            f"generated {scenario.scenario_id}: "
            f"frames={len(frames)} final_risk={final_level} "
            f"path={scenario_dir}"
        )


def main() -> None:
    args = parse_args()
    if args.command == "list":
        for scenario in SCENARIOS:
            print(
                f"{scenario.scenario_id}\t{scenario.risk_level}\t"
                f"{scenario.name}\t{scenario.abnormal_type}"
            )
        return
    asyncio.run(_run_generate(args))


if __name__ == "__main__":
    main()
