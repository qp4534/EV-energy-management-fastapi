# 백엔드 전달 계약

## 배포할 것

이 폴더 전체를 Docker 이미지에 복사한다. `models/hybrid_v1`의 두 `.joblib`,
`model_manifest.json`, `probability_calibration.json`은 한 세트이므로 하나라도
누락하면 안 된다.

## 배포하지 않는 것

- `robust_v3` 및 다른 연구/학습 후보
- AI-Hub·ORNL·Shenzhen 원시 데이터
- 학습 캐시·OOF 결과·노트북
- 팀원의 180초 onset 모델

## 필수 1Hz 입력

`voltage_v`, `temp_mean_c`, `temp_max_c`, `temp_delta_c`,
`temp_saturation_fraction`, `temp_saturation_all`

## 선택 BMS 입력

`raw_temp_max_c`, `raw_temp_mean_c`, `ambient_temp_c`, `pack_current_a`,
`cell_voltages_v`, `charging_gun_temperature_c`

선택 입력이 없다면 해당 물리 규칙만 적용하지 않는다. 값을 0이나 평균으로
채우면 안 된다. 충전건 온도는 충전 설비 관찰값이며 셀 온도가 아니다.

## API 출력

`sensor_health`, `ml_pattern_stage`, `ml_probabilities`,
`physical_rule_level`, `final_safety_alert`,
`charging_equipment_observation`, `reason_codes`, `history_seconds`
