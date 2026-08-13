from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from app.bms_demo import (
    _temperature_array,
    build_backend_payload,
    load_demo_dataset,
    model_route,
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
    assert "source_service_stage" not in payload
    assert "riskLevel" not in payload
    assert "finalRiskLevel" not in payload


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
