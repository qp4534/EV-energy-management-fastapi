from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from uuid import UUID

import pytest
import pandas as pd

from app.bms_demo import (
    _temperature_array,
    build_backend_payload,
    build_model_sample,
    generate_continuous_demo_rows,
    load_demo_dataset,
    model_route,
    parse_args,
    run_local_ml_demo,
    service_stage,
)


def test_service_stage_mapping_matches_deployed_contract() -> None:
    assert [service_stage(stage) for stage in range(1, 7)] == [0, 1, 1, 2, 2, 3]


def test_model_route_uses_30_and_120_sample_boundaries() -> None:
    assert model_route(29) == "warming_up"
    assert model_route(30) == "stage_30s"
    assert model_route(119) == "stage_30s"
    assert model_route(120) == "stage_120s"


def test_temperature_adapter_preserves_two_sensor_mean_and_delta() -> None:
    values = _temperature_array(40.0, 60.0)
    temperatures = [value / 10.0 for value in values]
    assert len(values) == 96
    assert sum(temperatures) / len(temperatures) == 50.0
    assert max(temperatures) == 60.0
    assert max(temperatures) - min(temperatures) == 20.0


def test_backend_payload_does_not_transmit_source_risk_label() -> None:
    row = {
        "source_service_stage": 3,
        "temperature_decic": [250] * 96,
        "voltage_mv": [3_800] * 96,
        "connector_temperature_decic": [250] * 3,
        "ambient_temperature_c": 25.0,
        "pack_current_a": None,
    }
    payload = build_backend_payload(
        row,
        session_id=UUID("11111111-1111-1111-1111-111111111111"),
        observed_at=datetime(2026, 8, 13, tzinfo=timezone.utc),
        sequence=0,
    )
    assert len(payload["temperatureDecic"]) == 96
    assert len(payload["voltageMv"]) == 96
    assert payload["sessionId"] == "11111111-1111-1111-1111-111111111111"
    assert "source_service_stage" not in payload
    assert "riskLevel" not in payload
    assert "finalRiskLevel" not in payload


def test_local_model_sample_matches_deployed_twin_adapter() -> None:
    row = {
        "source_service_stage": 3,
        "temperature_decic": [400, 600] * 48,
        "voltage_mv": [3_800] * 96,
        "connector_temperature_decic": [250] * 3,
        "ambient_temperature_c": 25.0,
        "pack_current_a": None,
    }
    sample = build_model_sample(row)
    assert sample["voltage_v"] == pytest.approx(3.8)
    assert sample["temp_mean_c"] == 50.0
    assert sample["temp_max_c"] == 60.0
    assert sample["temp_delta_c"] == 20.0
    assert "source_service_stage" not in sample


def test_committed_demo_dataset_has_expected_contract() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "demo"
        / "bms_hgb"
        / "aihub_holdout_20250912005_demo.jsonl"
    )
    rows = load_demo_dataset(path)
    assert len(rows) == 250
    assert [
        sum(row["source_service_stage"] == stage for row in rows)
        for stage in range(4)
    ] == [120, 50, 50, 30]


def test_replay_parser_requires_real_charging_session_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    session_id = "22222222-2222-2222-2222-222222222222"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bms_demo",
            "replay",
            "--dataset",
            str(tmp_path / "demo.jsonl"),
            "--backend-url",
            "https://backend.example",
            "--car-id",
            "11111111-1111-1111-1111-111111111111",
            "--charging-session-id",
            session_id,
            "--result",
            str(tmp_path / "result.jsonl"),
        ],
    )

    args = parse_args()

    assert args.charging_session_id == UUID(session_id)


def test_local_ml_demo_observes_all_four_deployed_model_stages(
    tmp_path: Path,
) -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "demo"
        / "bms_hgb"
        / "aihub_holdout_20250912005_demo.jsonl"
    )
    summary = run_local_ml_demo(
        load_demo_dataset(path),
        speed=0,
        print_every=1_000,
        result_path=tmp_path / "local-result.jsonl",
    )
    assert summary["observed_all_four_ml_stages"] is True
    assert [item["stage"] for item in summary["transitions"]] == [
        "warming_up",
        "normal",
        "caution",
        "warning",
        "emergency",
    ]
    assert summary["network_or_database_used"] is False


def test_continuous_generator_preserves_one_second_grid(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    frame = pd.DataFrame(
        {
            "experiment_id": ["E"] * 7,
            "elapsed_sec": [0, 120, 121, 122, 123, 124, 125],
            "voltage_v": [3.8, 3.8, 3.7, 3.6, 3.5, 3.0, 2.0],
            "surface_temp_c": [25, 30, 35, 45, 60, 80, 100],
            "positive_terminal_temp_c": [25, 31, 36, 46, 61, 81, 101],
            "ambient_temp_c": [23] * 7,
            "tr_stage": [1, 1, 2, 3, 4, 5, 6],
        }
    )
    frame.to_csv(source, index=False)
    rows = generate_continuous_demo_rows(
        source,
        experiment_id="E",
        warmup_before_caution_seconds=120,
        emergency_tail_seconds=1,
    )
    assert [row["sequence"] for row in rows] == list(range(len(rows)))
    assert [row["source_elapsed_sec"] for row in rows] == list(
        range(1, 126)
    )
    assert rows[-1]["source_service_stage"] == 3
    assert all(row["ambient_temperature_c"] is None for row in rows)
