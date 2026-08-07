from __future__ import annotations

from app.scenario_catalog import (
    NORMAL_SCENARIO_ID,
    SCENARIO_BY_ID,
    SCENARIOS,
    scenario_for_abnormal_type,
)


def test_catalog_has_exactly_ten_scenarios() -> None:
    assert len(SCENARIOS) == 10
    assert len(SCENARIO_BY_ID) == 10


def test_catalog_risk_level_distribution() -> None:
    assert {scenario.risk_level for scenario in SCENARIOS} == {0, 1, 2, 3}
    assert sum(scenario.risk_level == 0 for scenario in SCENARIOS) == 1
    assert sum(scenario.risk_level == 1 for scenario in SCENARIOS) == 3
    assert sum(scenario.risk_level == 2 for scenario in SCENARIOS) == 3
    assert sum(scenario.risk_level == 3 for scenario in SCENARIOS) == 3


def test_catalog_ids_are_unique_and_url_safe() -> None:
    ids = [scenario.scenario_id for scenario in SCENARIOS]
    assert len(ids) == len(set(ids))
    assert all(scenario_id.replace("_", "").isalnum() for scenario_id in ids)


def test_abnormal_type_mapping_matches_anomaly_logs() -> None:
    cases = {
        "커넥터 국부 과열": "connector_local_overheat",
        "배터리 임계온도 초과": "battery_over_temp",
        "열폭주 위험": "thermal_runaway_risk",
        "셀 전압 불균형": "cell_voltage_imbalance",
        "배터리 과열 징후": "battery_overheat_sign",
        "급격한 온도 상승": "rapid_temp_rise",
        "커넥터 온도 상승": "connector_temp_rise",
        "셀 전압 편차 증가": "cell_voltage_deviation",
        "충전 전류 변동": "charging_current_fluctuation",
    }
    for abnormal_type, expected_id in cases.items():
        assert (
            scenario_for_abnormal_type(abnormal_type).scenario_id
            == expected_id
        )


def test_missing_or_normal_logs_map_to_normal() -> None:
    for abnormal_type in (
        None,
        "정상",
        "배터리 상태 정상",
        "열화상 정상",
        "충전 상태 정상",
        "알 수 없는 유형",
    ):
        assert (
            scenario_for_abnormal_type(abnormal_type).scenario_id
            == NORMAL_SCENARIO_ID
        )
