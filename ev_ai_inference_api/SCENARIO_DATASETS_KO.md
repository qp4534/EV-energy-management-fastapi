# 10종 시나리오 디지털 트윈 데이터셋

실시간 디지털 트윈용 1시간 데이터셋(정상 1 + 이상 9)을 미리 생성하고,
140대 차량이 저장된 데이터를 반복 재생하는 구조입니다.

## 시나리오 10종

| 시나리오 ID | 이름 | 위험도 |
|---|---|---:|
| normal | 정상 | 0 |
| connector_local_overheat | 커넥터 국부 과열 | 3 |
| battery_over_temp | 배터리 임계온도 초과 | 3 |
| thermal_runaway_risk | 열폭주 위험 | 3 |
| cell_voltage_imbalance | 셀 전압 불균형 | 2 |
| battery_overheat_sign | 배터리 과열 징후 | 2 |
| rapid_temp_rise | 급격한 온도 상승 | 2 |
| connector_temp_rise | 커넥터 온도 상승 | 1 |
| cell_voltage_deviation | 셀 전압 편차 증가 | 1 |
| charging_current_fluctuation | 충전 전류 변동 | 1 |

## 데이터 생성

각 시나리오의 1Hz 센서 프레임을 BMS 하이브리드 모델에 통과시켜
최종 `TwinFrame`을 만들고 `runtime/scenarios/<scenario_id>/frames.jsonl`로 저장합니다.

```powershell
python -m app.scenario_generator list
python -m app.scenario_generator generate --scenario all --frame-count 3600
```

열화상 워커가 설정되어 있으면 5초 간격으로 열화상을 렌더링하고 추론합니다.

```powershell
python -m app.scenario_generator generate --scenario all --with-thermal
```

## 실시간 재생

차량 목록을 RDS(`CAR` + 최신 `ANOMALY_LOGS`)에서 읽어 시나리오에 매핑한 뒤,
1초마다 미리 생성된 프레임을 Redis에 publish합니다. AI 재추론은 하지 않습니다.

```powershell
python -m app.scenario_replay replay --scenario-dir runtime/scenarios
```

DB 없이 로컬 테스트하려면 차량 할당 파일을 사용합니다.

```json
[
  {
    "vehicle_id": "car-uuid-001",
    "car_number": "11가6762",
    "model": "GV60",
    "scenario_id": "normal",
    "offset_seconds": 0
  }
]
```

```powershell
python -m app.scenario_replay replay `
  --scenario-dir runtime/scenarios `
  --assignments-file runtime/assignments.json `
  --speed 60
```

시뮬레이션 차량은 `publish_live_only()`를 사용해 Redis latest/pubsub만 갱신하므로
prebuffer/persist 저장을 하지 않습니다.

## 참고

- `app/scenario_catalog.py`: 10종 시나리오 정의 및 ANOMALY_LOGS 이상 유형 매핑
- `app/scenario_generator.py`: 1시간 데이터 생성기
- `app/scenario_replay.py`: 140대 재생 워커
- `app/core/twin_redis.py`: `publish_live_only` 저장 생략 publish
