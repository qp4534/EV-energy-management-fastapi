from __future__ import annotations

from dataclasses import dataclass


NORMAL_SCENARIO_ID = "normal"


@dataclass(frozen=True)
class ScenarioDefinition:
    """One real-time digital-twin scenario backed by a pre-generated 1-hour dataset."""

    scenario_id: str
    name: str
    abnormal_type: str
    risk_level: int
    incident_type: str | None
    source_type: str
    description: str


SCENARIOS: tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition(
        scenario_id=NORMAL_SCENARIO_ID,
        name="정상",
        abnormal_type="정상",
        risk_level=0,
        incident_type=None,
        source_type="CHARGING",
        description="충전 전력과 배터리·커넥터 온도가 안정적으로 유지되는 기준 시나리오",
    ),
    ScenarioDefinition(
        scenario_id="connector_local_overheat",
        name="커넥터 국부 과열",
        abnormal_type="커넥터 국부 과열",
        risk_level=3,
        incident_type="connector",
        source_type="THERMAL",
        description="충전 커넥터 접점이 80°C 이상으로 국부 과열되는 긴급 시나리오",
    ),
    ScenarioDefinition(
        scenario_id="battery_over_temp",
        name="배터리 임계온도 초과",
        abnormal_type="배터리 임계온도 초과",
        risk_level=3,
        incident_type="battery",
        source_type="BMS",
        description="배터리 셀 온도가 임계값을 넘어 긴급 알림이 발생하는 시나리오",
    ),
    ScenarioDefinition(
        scenario_id="thermal_runaway_risk",
        name="열폭주 위험",
        abnormal_type="열폭주 위험",
        risk_level=3,
        incident_type="battery",
        source_type="AI_MODEL",
        description="열화상·센서 융합 모델이 열폭주 위험으로 판단하는 시나리오",
    ),
    ScenarioDefinition(
        scenario_id="cell_voltage_imbalance",
        name="셀 전압 불균형",
        abnormal_type="셀 전압 불균형",
        risk_level=2,
        incident_type="battery",
        source_type="BMS",
        description="특정 셀 전압이 크게 낮아져 불균형이 심화되는 경고 시나리오",
    ),
    ScenarioDefinition(
        scenario_id="battery_overheat_sign",
        name="배터리 과열 징후",
        abnormal_type="배터리 과열 징후",
        risk_level=2,
        incident_type="battery",
        source_type="THERMAL",
        description="배터리 국부 온도가 상승하며 과열 전조가 나타나는 경고 시나리오",
    ),
    ScenarioDefinition(
        scenario_id="rapid_temp_rise",
        name="급격한 온도 상승",
        abnormal_type="급격한 온도 상승",
        risk_level=2,
        incident_type="battery",
        source_type="AI_MODEL",
        description="후반부에 셀 온도가 급격히 상승하는 경고 시나리오",
    ),
    ScenarioDefinition(
        scenario_id="connector_temp_rise",
        name="커넥터 온도 상승",
        abnormal_type="커넥터 온도 상승",
        risk_level=1,
        incident_type="connector",
        source_type="THERMAL",
        description="충전 커넥터 온도가 주의 수준까지 상승하는 시나리오",
    ),
    ScenarioDefinition(
        scenario_id="cell_voltage_deviation",
        name="셀 전압 편차 증가",
        abnormal_type="셀 전압 편차 증가",
        risk_level=1,
        incident_type="battery",
        source_type="BMS",
        description="셀 전압 편차가 주의 수준으로 증가하는 시나리오",
    ),
    ScenarioDefinition(
        scenario_id="charging_current_fluctuation",
        name="충전 전류 변동",
        abnormal_type="충전 전류 변동",
        risk_level=1,
        incident_type="general",
        source_type="CHARGING",
        description="충전 전류가 크게 출렁이며 주의가 필요한 시나리오",
    ),
)

SCENARIO_BY_ID: dict[str, ScenarioDefinition] = {
    scenario.scenario_id: scenario for scenario in SCENARIOS
}

NORMAL_ABNORMAL_TYPES = frozenset(
    {"정상", "배터리 상태 정상", "열화상 정상", "충전 상태 정상"}
)

ABNORMAL_TYPE_TO_SCENARIO_ID: dict[str, str] = {
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


def scenario_for_abnormal_type(abnormal_type: str | None) -> ScenarioDefinition:
    """Map an ANOMALY_LOGS abnormal_type to a scenario; missing/unknown -> normal."""

    if abnormal_type is None or abnormal_type in NORMAL_ABNORMAL_TYPES:
        return SCENARIO_BY_ID[NORMAL_SCENARIO_ID]
    scenario_id = ABNORMAL_TYPE_TO_SCENARIO_ID.get(abnormal_type)
    if scenario_id is None:
        return SCENARIO_BY_ID[NORMAL_SCENARIO_ID]
    return SCENARIO_BY_ID[scenario_id]
