import sys
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from conftest import sample

def test_fusion_blocks_temperature_only_emergency_and_allows_corroboration():
    with TestClient(app) as client:
        hot_only = client.post("/v1/vehicles/H/samples", json=sample(1, raw_temp_max_c=85, raw_temp_mean_c=30)).json()
        assert hot_only["final_safety_alert"] != "emergency"
        result = client.post("/v1/vehicles/H/samples", json=sample(2, raw_temp_max_c=85, raw_temp_mean_c=65)).json()
        assert result["final_safety_alert"] == "emergency"
        gun = client.post("/v1/vehicles/G/samples", json=sample(1, charging_gun_temperature_c=100)).json()
        assert gun["final_safety_alert"] != "emergency"


def test_fusion_does_not_count_over_ambient_signal_twice():
    with TestClient(app) as client:
        result = client.post(
            "/v1/vehicles/OVER-AMBIENT/samples",
            json=sample(
                1,
                temp_mean_c=55,
                temp_max_c=65,
                raw_temp_mean_c=55,
                raw_temp_max_c=65,
                ambient_temp_c=23,
            ),
        ).json()
        assert result["physical_rule_level"] == "emergency"
        assert result["final_safety_alert"] == "warning"
        assert "physical_emergency_candidate_not_corroborated" in result["reason_codes"]


def test_fusion_allows_two_distinct_physical_evidence_groups():
    with TestClient(app) as client:
        result = client.post(
            "/v1/vehicles/LOCAL-MEAN/samples",
            json=sample(
                1,
                temp_mean_c=65,
                temp_max_c=85,
                raw_temp_mean_c=65,
                raw_temp_max_c=85,
                ambient_temp_c=23,
            ),
        ).json()
        assert result["final_safety_alert"] == "emergency"
        assert "corroborated_physical_emergency_v21" in result["reason_codes"]
