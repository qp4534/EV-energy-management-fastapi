from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd


DEFAULT_EXPERIMENT_ID = "20250912005"
DEFAULT_STAGE_COUNTS = {0: 120, 1: 50, 2: 50, 3: 30}
RISK_NAMES = {0: "normal", 1: "caution", 2: "warning", 3: "emergency"}
RISK_LEVELS = {name: level for level, name in RISK_NAMES.items()}


def service_stage(raw_stage: int) -> int:
    if raw_stage == 1:
        return 0
    if raw_stage in {2, 3}:
        return 1
    if raw_stage in {4, 5}:
        return 2
    if raw_stage == 6:
        return 3
    raise ValueError(f"unsupported AI-Hub stage: {raw_stage}")


def model_route(sample_count: int) -> str:
    if sample_count < 30:
        return "warming_up"
    if sample_count < 120:
        return "stage_30s"
    return "stage_120s"


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("stage sample count must be positive")
    if length < count:
        raise ValueError(f"stage has {length} rows but {count} are required")
    if count == 1:
        return [length - 1]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def _temperature_array(surface_c: float, terminal_c: float) -> list[int]:
    values = []
    for cell_index in range(96):
        source = surface_c if cell_index % 2 == 0 else terminal_c
        values.append(round(max(-40.0, min(150.0, source)) * 10.0))
    return values


def _voltage_array(voltage_v: float) -> list[int]:
    value = round(max(0.0, min(6.0, voltage_v)) * 1_000.0)
    return [value] * 96


def _connector_array(ambient_c: float) -> list[int]:
    value = round(max(-40.0, min(150.0, ambient_c)) * 10.0)
    return [value, value, value]


def generate_demo_rows(
    source_csv: Path,
    *,
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    stage_counts: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    counts = stage_counts or DEFAULT_STAGE_COUNTS
    frame = pd.read_csv(
        source_csv,
        dtype={"experiment_id": "string"},
        usecols=[
            "experiment_id",
            "elapsed_sec",
            "voltage_v",
            "surface_temp_c",
            "positive_terminal_temp_c",
            "ambient_temp_c",
            "tr_stage",
        ],
    )
    frame = frame[frame["experiment_id"] == str(experiment_id)].copy()
    if frame.empty:
        raise ValueError(f"experiment {experiment_id} was not found in {source_csv}")
    frame = frame.sort_values("elapsed_sec").reset_index(drop=True)
    frame["service_stage"] = frame["tr_stage"].map(
        lambda value: service_stage(int(value))
    )

    rows: list[dict[str, Any]] = []
    sequence = 0
    for target_stage in range(4):
        stage_frame = frame[frame["service_stage"] == target_stage].reset_index(
            drop=True
        )
        for source_index in _evenly_spaced_indices(
            len(stage_frame), counts[target_stage]
        ):
            source = stage_frame.iloc[source_index]
            rows.append(
                {
                    "sequence": sequence,
                    "source_experiment_id": str(experiment_id),
                    "source_elapsed_sec": float(source["elapsed_sec"]),
                    "source_raw_stage": int(source["tr_stage"]),
                    "source_service_stage": target_stage,
                    "temperature_decic": _temperature_array(
                        float(source["surface_temp_c"]),
                        float(source["positive_terminal_temp_c"]),
                    ),
                    "voltage_mv": _voltage_array(float(source["voltage_v"])),
                    "connector_temperature_decic": _connector_array(
                        float(source["ambient_temp_c"])
                    ),
                    "ambient_temperature_c": float(source["ambient_temp_c"]),
                    "pack_current_a": None,
                }
            )
            sequence += 1
    return rows


def generate_continuous_demo_rows(
    source_csv: Path,
    *,
    experiment_id: str,
    warmup_before_caution_seconds: int = 120,
    emergency_tail_seconds: int = 1,
) -> list[dict[str, Any]]:
    """Build a causal 1 Hz excerpt without stage-wise time compression."""

    if warmup_before_caution_seconds < 120:
        raise ValueError("continuous demo requires at least 120 warm-up seconds")
    if emergency_tail_seconds < 1:
        raise ValueError("emergency_tail_seconds must be positive")
    frame = pd.read_csv(
        source_csv,
        dtype={"experiment_id": "string"},
        usecols=[
            "experiment_id",
            "elapsed_sec",
            "voltage_v",
            "surface_temp_c",
            "positive_terminal_temp_c",
            "ambient_temp_c",
            "tr_stage",
        ],
    )
    frame = frame[frame["experiment_id"] == str(experiment_id)].copy()
    if frame.empty:
        raise ValueError(f"experiment {experiment_id} was not found in {source_csv}")
    frame = (
        frame.sort_values("elapsed_sec")
        .drop_duplicates("elapsed_sec", keep="last")
        .reset_index(drop=True)
    )
    frame["service_stage"] = frame["tr_stage"].map(
        lambda value: service_stage(int(value))
    )
    if not all((frame["service_stage"] == stage).any() for stage in range(4)):
        raise ValueError("continuous demo experiment must contain all four service stages")

    caution_start = float(
        frame.loc[frame["service_stage"] == 1, "elapsed_sec"].iloc[0]
    )
    emergency_start = float(
        frame.loc[frame["service_stage"] == 3, "elapsed_sec"].iloc[0]
    )
    start = max(
        float(frame["elapsed_sec"].min()),
        caution_start - warmup_before_caution_seconds,
    )
    end = min(
        float(frame["elapsed_sec"].max()),
        emergency_start + emergency_tail_seconds,
    )
    grid = np.arange(np.ceil(start), np.floor(end) + 1.0, 1.0)
    elapsed = frame["elapsed_sec"].to_numpy(dtype=float)
    interpolated = {
        column: np.interp(grid, elapsed, frame[column].to_numpy(dtype=float))
        for column in (
            "voltage_v",
            "surface_temp_c",
            "positive_terminal_temp_c",
            "ambient_temp_c",
        )
    }
    source_indices = np.searchsorted(elapsed, grid, side="right") - 1
    source_indices = np.clip(source_indices, 0, len(frame) - 1)

    rows: list[dict[str, Any]] = []
    for sequence, (source_elapsed, source_index) in enumerate(
        zip(grid, source_indices, strict=True)
    ):
        source = frame.iloc[int(source_index)]
        rows.append(
            {
                "sequence": sequence,
                "source_experiment_id": str(experiment_id),
                "source_elapsed_sec": float(source_elapsed),
                "source_raw_stage": int(source["tr_stage"]),
                "source_service_stage": int(source["service_stage"]),
                "source_interpolated_to_1hz": bool(
                    not np.isclose(source_elapsed, float(source["elapsed_sec"]))
                ),
                "temperature_decic": _temperature_array(
                    float(interpolated["surface_temp_c"][sequence]),
                    float(interpolated["positive_terminal_temp_c"][sequence]),
                ),
                "voltage_mv": _voltage_array(
                    float(interpolated["voltage_v"][sequence])
                ),
                "connector_temperature_decic": _connector_array(
                    float(interpolated["ambient_temp_c"][sequence])
                ),
                # AI-Hub ambient_temp_c is a laboratory-chamber signal.  It is
                # not guaranteed to mean vehicle pack ambient/coolant
                # temperature, so the optional vehicle physical-rule input is
                # deliberately unavailable rather than semantically invented.
                "ambient_temperature_c": None,
                "pack_current_a": None,
            }
        )
    return rows


def write_demo_dataset(
    rows: list[dict[str, Any]],
    output_path: Path,
    manifest_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")

    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    stage_counts = {
        RISK_NAMES[stage]: sum(
            row["source_service_stage"] == stage for row in rows
        )
        for stage in range(4)
    }
    manifest = {
        "dataset_id": "bms-hgb-demo-aihub-holdout-20250912005-v1",
        "purpose": "time-compressed integration demonstration only",
        "source": {
            "dataset": "AI-Hub battery thermal-runaway controlled experiment CSV",
            "experiment_id": rows[0]["source_experiment_id"],
            "locked_split": "test",
        },
        "rows": len(rows),
        "logical_frequency_hz": 1,
        "service_stage_source_counts": stage_counts,
        "adapter": {
            "temperature": (
                "48 cells use surface temperature and 48 cells use positive-terminal "
                "temperature; values are clipped to the deployed -40..150 C contract"
            ),
            "voltage": "the AI-Hub cell voltage is repeated across 96 cells",
            "connector_temperature": "ambient temperature is used to avoid inventing connector abuse",
            "risk_labels_transmitted_to_backend": False,
        },
        "limitations": [
            "The 96-cell layout is a deterministic presentation adapter, not measured vehicle cell data.",
            "Rows are evenly sampled within each service stage, so elapsed time is compressed.",
            "The dataset must not be used as new validation evidence or a real-vehicle claim.",
        ],
        "sha256": digest,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_continuous_demo_dataset(
    rows: list[dict[str, Any]],
    output_path: Path,
    manifest_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    experiment_id = rows[0]["source_experiment_id"]
    manifest = {
        "dataset_id": f"bms-hgb-continuous-validation-{experiment_id}-v1",
        "purpose": "causal time-compressed presentation replay only",
        "source": {
            "dataset": "AI-Hub battery thermal-runaway controlled experiment CSV",
            "experiment_id": experiment_id,
            "locked_split": "validation",
            "source_elapsed_start": rows[0]["source_elapsed_sec"],
            "source_elapsed_end": rows[-1]["source_elapsed_sec"],
        },
        "rows": len(rows),
        "logical_frequency_hz": 1,
        "service_stage_source_counts": {
            RISK_NAMES[stage]: sum(
                row["source_service_stage"] == stage for row in rows
            )
            for stage in range(4)
        },
        "selection": {
            "pre_screened_validation_candidates": [
                "20250825004",
                "20250811005",
                "20250820002",
            ],
            "criteria": [
                "all four HGB stages appear in monotonic order",
                "short presentation duration",
                "minimize rows where Safety Fusion exceeds HGB by two levels",
            ],
            "selected_runtime_audit": {
                "logical_rows": 586,
                "fusion_minus_hgb_ge_2_rows": 4,
                "hgb_stage_regression": False,
            },
        },
        "adapter": {
            "temperature": "48 surface-temperature cells plus 48 positive-terminal-temperature cells",
            "voltage": "AI-Hub cell voltage repeated across 96 cells",
            "missing_source_seconds": "linear interpolation to a causal 1 Hz grid",
            "ambient_temperature_c": (
                "not transmitted: AI-Hub chamber ambient is not guaranteed to be "
                "semantically equivalent to vehicle pack ambient/coolant temperature"
            ),
            "connector_temperature": (
                "AI-Hub chamber ambient retained only as a neutral visualization "
                "placeholder and never used as cell temperature"
            ),
            "risk_labels_transmitted_to_backend": False,
        },
        "limitations": [
            "This is a selected validation experiment for presentation, not new performance evidence.",
            "The 96-cell layout is an adapter and not measured vehicle cell data.",
            "The source label is retained only for offline review and never sent to the model or Backend.",
        ],
        "sha256": digest,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_demo_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if len(row.get("temperature_decic", [])) != 96:
                raise ValueError(f"line {line_number}: temperature_decic must have 96 values")
            if len(row.get("voltage_mv", [])) != 96:
                raise ValueError(f"line {line_number}: voltage_mv must have 96 values")
            if len(row.get("connector_temperature_decic", [])) != 3:
                raise ValueError(
                    f"line {line_number}: connector_temperature_decic must have 3 values"
                )
            rows.append(row)
    if not rows:
        raise ValueError("demo dataset is empty")
    return rows


def build_backend_payload(
    row: dict[str, Any],
    *,
    session_id: UUID,
    observed_at: datetime,
    sequence: int,
) -> dict[str, Any]:
    payload = {
        "sessionId": str(session_id),
        "observedAt": observed_at.isoformat(),
        "sequence": sequence,
        "temperatureDecic": list(row["temperature_decic"]),
        "voltageMv": list(row["voltage_mv"]),
        "connectorTemperatureDecic": list(row["connector_temperature_decic"]),
        "ambientTemperatureC": row.get("ambient_temperature_c"),
        "packCurrentA": row.get("pack_current_a"),
    }
    return payload


def build_model_sample(row: dict[str, Any]) -> dict[str, Any]:
    """Apply the same BMS-array adapter used by the deployed Twin service."""

    temperatures_c = [value / 10.0 for value in row["temperature_decic"]]
    voltages_v = [value / 1_000.0 for value in row["voltage_mv"]]
    maximum = max(temperatures_c)
    minimum = min(temperatures_c)
    saturated = sum(value >= 150.0 for value in temperatures_c)
    connector_temperatures_c = [
        value / 10.0 for value in row["connector_temperature_decic"]
    ]
    return {
        "voltage_v": sum(voltages_v) / len(voltages_v),
        "temp_mean_c": sum(temperatures_c) / len(temperatures_c),
        "temp_max_c": maximum,
        "temp_delta_c": maximum - minimum,
        "temp_saturation_fraction": saturated / len(temperatures_c),
        "temp_saturation_all": saturated == len(temperatures_c),
        "raw_temp_max_c": maximum,
        "raw_temp_mean_c": sum(temperatures_c) / len(temperatures_c),
        "ambient_temp_c": row.get("ambient_temperature_c"),
        "pack_current_a": row.get("pack_current_a"),
        "cell_voltages_v": voltages_v,
        "temperature_decic": list(row["temperature_decic"]),
        "connector_temperature_decic": list(
            row["connector_temperature_decic"]
        ),
        "charging_gun_temperature_c": max(connector_temperatures_c),
    }


def _deployed_supervisor() -> Any:
    """Load the checksum-verified model bundle used by the FastAPI service."""

    from app.core.config import Settings, validate_bundle

    settings = Settings.load()
    validate_bundle(settings)
    from hybrid_safety_supervisor_v21 import HybridSafetySupervisorV21

    return HybridSafetySupervisorV21(
        settings.bundle_dir / "models" / "hybrid_v1",
        settings.bundle_dir / "config" / "safety_policy.v2.json",
    )


def run_local_ml_demo(
    rows: list[dict[str, Any]],
    *,
    speed: float,
    result_path: Path,
    print_every: int = 10,
    stop_on_ml_level: int | None = None,
    supervisor: Any | None = None,
) -> dict[str, Any]:
    """Replay only the deployed Hybrid HGB path without network or persistence.

    Physical and final levels are recorded for audit, but the console foregrounds
    the ML stage and calibrated class probabilities. Source labels are never
    included in the model sample.
    """

    if speed < 0:
        raise ValueError("speed must be zero or positive")
    if print_every <= 0:
        raise ValueError("print_every must be positive")
    if stop_on_ml_level is not None and stop_on_ml_level not in RISK_NAMES:
        raise ValueError("stop_on_ml_level must be between 0 and 3")

    runtime = supervisor or _deployed_supervisor()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    last_stage: str | None | object = object()
    transitions: list[dict[str, Any]] = []
    counts = {"warming_up": 0, **{name: 0 for name in RISK_NAMES.values()}}
    processed = 0

    print("=" * 78)
    print("BMS HYBRID HGB - LOCAL MODEL DEMONSTRATION")
    print("same checksum-verified bundle and 30s/120s routing as deployed FastAPI")
    print("levels: 0=normal, 1=caution, 2=warning, 3=emergency")
    print("=" * 78)

    with result_path.open("w", encoding="utf-8", newline="\n") as log:
        for sequence, row in enumerate(rows):
            result = runtime.push(build_model_sample(row), float(sequence))
            stage = result.ml_pattern_stage
            route = model_route(sequence + 1)
            probabilities = result.ml_probabilities
            display_stage = stage or "warming_up"
            counts[display_stage] += 1

            record = {
                "sequence": sequence,
                "model_route": route,
                "source_experiment_id": row.get("source_experiment_id"),
                "source_elapsed_sec": row.get("source_elapsed_sec"),
                "source_service_stage_review_only": row.get(
                    "source_service_stage"
                ),
                "ml_risk_level": (
                    None if stage is None else RISK_LEVELS[stage]
                ),
                "ml_pattern_stage": stage,
                "current_stage_probabilities": probabilities,
                "physics_risk_level": RISK_LEVELS[result.physical_rule_level],
                "final_risk_level": RISK_LEVELS[result.final_safety_alert],
            }
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            processed += 1

            changed = stage != last_stage
            if changed:
                transitions.append(
                    {
                        "sequence": sequence,
                        "route": route,
                        "stage": display_stage,
                        "probabilities": probabilities,
                    }
                )
            if changed or sequence % print_every == 0:
                if probabilities is None:
                    probability_text = "probabilities=warming-up"
                else:
                    probability_text = " ".join(
                        f"{name[:1].upper()}={probabilities[name]:.3f}"
                        for name in RISK_NAMES.values()
                    )
                print(
                    f"seq={sequence:03d} route={route:<10} "
                    f"HGB={display_stage:<10} {probability_text}"
                )
            last_stage = stage

            if (
                stop_on_ml_level is not None
                and stage is not None
                and RISK_LEVELS[stage] == stop_on_ml_level
            ):
                break
            if speed > 0 and sequence + 1 < len(rows):
                time.sleep(1.0 / speed)

    observed = {
        transition["stage"]
        for transition in transitions
        if transition["stage"] != "warming_up"
    }
    summary = {
        "processed_rows": processed,
        "observed_all_four_ml_stages": observed == set(RISK_NAMES.values()),
        "stage_counts": counts,
        "transitions": transitions,
        "result_path": str(result_path),
        "network_or_database_used": False,
    }
    print("-" * 78)
    print(
        "ML transitions: "
        + " -> ".join(
            f"{item['stage']}@{item['sequence']}" for item in transitions
        )
    )
    print(f"all four ML stages observed: {summary['observed_all_four_ml_stages']}")
    print(f"result: {result_path}")
    return summary


def replay_via_backend(
    rows: list[dict[str, Any]],
    *,
    backend_url: str,
    car_id: UUID,
    charging_session_id: UUID,
    jwt_token: str,
    speed: float,
    result_path: Path,
    start_at: datetime | None = None,
    stop_on_level: int | None = None,
) -> dict[str, Any]:
    import httpx

    if speed <= 0:
        raise ValueError("speed must be positive")
    if stop_on_level is not None and stop_on_level not in RISK_NAMES:
        raise ValueError("stop_on_level must be between 0 and 3")
    # The shared RDS schema treats session_id as a foreign key to
    # CHARGING_SESSION.  Reusing an arbitrary inference-only UUID works while
    # the result is normal, but fails on the first anomalous persistence write.
    # A replay therefore has to use a real charging session created for the
    # demo car through Spring Backend.
    session_id = charging_session_id
    logical_start = start_at or datetime.now(timezone.utc).replace(microsecond=0)
    endpoint = (
        f"{backend_url.rstrip('/')}/api/twin-frames/cars/{car_id}/bms-samples"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    last_levels: tuple[Any, Any, Any] | None = None
    sent = 0

    with httpx.Client(
        headers={"Authorization": f"Bearer {jwt_token}"},
        timeout=10.0,
    ) as client, result_path.open("w", encoding="utf-8", newline="\n") as log:
        for sequence, row in enumerate(rows):
            observed_at = logical_start + timedelta(seconds=sequence)
            payload = build_backend_payload(
                row,
                session_id=session_id,
                observed_at=observed_at,
                sequence=sequence,
            )
            response = client.post(endpoint, json=payload)
            if response.status_code in {401, 403}:
                raise RuntimeError(
                    "Backend rejected the JWT or this user has no access to the demo car"
                )
            response.raise_for_status()
            inference = response.json()
            levels = (
                inference.get("ml_risk_level"),
                inference.get("physics_risk_level"),
                inference.get("final_risk_level"),
            )
            record = {
                "sequence": sequence,
                "observed_at": observed_at.isoformat(),
                "model_route": model_route(sequence + 1),
                "source_experiment_id": row.get("source_experiment_id"),
                "source_elapsed_sec": row.get("source_elapsed_sec"),
                "source_service_stage_review_only": row.get(
                    "source_service_stage"
                ),
                "ml_risk_level": levels[0],
                "physics_risk_level": levels[1],
                "final_risk_level": levels[2],
                "anomaly_id": inference.get("anomaly_id"),
            }
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            sent += 1

            if levels != last_levels or sequence % 30 == 0:
                print(
                    f"seq={sequence:03d} route={record['model_route']} "
                    f"ml={levels[0]} physics={levels[1]} final={levels[2]}"
                )
                last_levels = levels
            if stop_on_level is not None and levels[2] == stop_on_level:
                break
            if sequence + 1 < len(rows):
                time.sleep(1.0 / speed)

    return {
        "session_id": str(session_id),
        "sent_rows": sent,
        "result_path": str(result_path),
    }


def _parse_start_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("start-at must include a timezone offset")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and replay a Hybrid HGB demonstration through Spring Backend"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate")
    generate.add_argument("--source-csv", type=Path, required=True)
    generate.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--manifest", type=Path, required=True)

    continuous = commands.add_parser(
        "generate-continuous",
        help="generate a causal 1 Hz validation-experiment presentation replay",
    )
    continuous.add_argument("--source-csv", type=Path, required=True)
    continuous.add_argument("--experiment-id", default="20250825004")
    continuous.add_argument("--warmup-seconds", type=int, default=120)
    continuous.add_argument("--emergency-tail-seconds", type=int, default=1)
    continuous.add_argument("--output", type=Path, required=True)
    continuous.add_argument("--manifest", type=Path, required=True)

    local_ml = commands.add_parser(
        "local-ml",
        help="demonstrate the deployed Hybrid HGB locally without Backend/RDS",
    )
    local_ml.add_argument("--dataset", type=Path, required=True)
    local_ml.add_argument(
        "--speed",
        type=float,
        default=5.0,
        help="logical rows per real second; use 0 for an immediate verification run",
    )
    local_ml.add_argument("--print-every", type=int, default=10)
    local_ml.add_argument(
        "--result",
        type=Path,
        default=Path("runtime/bms_hgb_local_ml_result.jsonl"),
    )
    local_ml.add_argument("--stop-on-ml-level", type=int, default=None)

    replay = commands.add_parser("replay")
    replay.add_argument("--dataset", type=Path, required=True)
    replay.add_argument("--backend-url", required=True)
    replay.add_argument("--car-id", type=UUID, required=True)
    replay.add_argument(
        "--charging-session-id",
        type=UUID,
        required=True,
        help="Existing CHARGING_SESSION.session_id for the demo car",
    )
    replay.add_argument("--speed", type=float, default=1.0)
    replay.add_argument("--start-at", type=_parse_start_at, default=None)
    replay.add_argument("--result", type=Path, required=True)
    replay.add_argument("--stop-on-level", type=int, default=None)
    replay.add_argument("--jwt-env", default="BMS_DEMO_JWT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        rows = generate_demo_rows(
            args.source_csv,
            experiment_id=args.experiment_id,
        )
        write_demo_dataset(rows, args.output, args.manifest)
        print(f"generated {len(rows)} logical 1 Hz rows: {args.output}")
        return

    if args.command == "generate-continuous":
        rows = generate_continuous_demo_rows(
            args.source_csv,
            experiment_id=args.experiment_id,
            warmup_before_caution_seconds=args.warmup_seconds,
            emergency_tail_seconds=args.emergency_tail_seconds,
        )
        write_continuous_demo_dataset(rows, args.output, args.manifest)
        print(f"generated {len(rows)} continuous logical 1 Hz rows: {args.output}")
        return

    if args.command == "local-ml":
        summary = run_local_ml_demo(
            load_demo_dataset(args.dataset),
            speed=args.speed,
            result_path=args.result,
            print_every=args.print_every,
            stop_on_ml_level=args.stop_on_ml_level,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    jwt_token = os.getenv(args.jwt_env, "").strip()
    if not jwt_token:
        raise SystemExit(
            f"{args.jwt_env} is empty; log in to Spring Backend and set the JWT in this environment"
        )
    result = replay_via_backend(
        load_demo_dataset(args.dataset),
        backend_url=args.backend_url,
        car_id=args.car_id,
        charging_session_id=args.charging_session_id,
        jwt_token=jwt_token,
        speed=args.speed,
        result_path=args.result,
        start_at=args.start_at,
        stop_on_level=args.stop_on_level,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
