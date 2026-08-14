# BMS Hybrid HGB 시연 구현 리뷰

## 결론

기존 RDS 대문자 `TWIN_FRAMES` 750행은 모두 합성 데이터이고
`model_input`이 비어 있어 현재 Hybrid HGB 시연 근거로 채택하지 않았다. 소문자
`twin_frames` 97,200행도 미리 계산된 3D Twin 재생 데이터이며 각 시나리오의
`max_ml`이 0이므로 모델 재추론 시연에 사용하지 않았다.

새 시연은 AI-Hub 잠금 test 실험 `20250912005`의 원시 센서 흐름을 96셀 입력으로
변환한 뒤 Spring Backend를 통해 배포 FastAPI에 다시 전달한다. 요청 본문에는
정답 단계나 사전 계산 위험등급이 없다.

## 변경 사항

- `app/bms_demo.py`
  - AI-Hub CSV에서 발표용 250초 JSONL 생성
  - 96셀·3커넥터 입력 계약 검증
  - Spring Backend JWT API로 순차 전송
  - 실제 `CHARGING_SESSION.session_id`를 필수 입력으로 받아 RDS 외래키 정합성 유지
  - ML·물리·최종 위험도와 `anomaly_id` 결과 JSONL 기록
- `demo/bms_hgb/aihub_holdout_20250912005_demo.jsonl`
  - 정상 120, 주의 50, 경고 50, 긴급 30 논리초
- `demo/bms_hgb/manifest.json`
  - 데이터 출처·해시·어댑터·제약 기록
- `tests/test_bms_demo.py`
  - 4단계 매핑, 30/120초 라우팅, 센서 배열, 라벨 비전송 검사

## 배포 번들 오프라인 dry-run

운영 FastAPI Pod에 포함된 동일 `HybridSafetySupervisorV2`와 HGB joblib을 사용했다.
Redis·RDS·HTTP를 사용하지 않았으며 센서 JSONL만 순서대로 입력했다.

| 구간 | Hybrid HGB 출력 |
|---|---|
| sequence 0~28 | warming_up |
| sequence 29~121 | 정상 |
| sequence 122~170 | 주의 |
| sequence 171~208 | 경고 |
| sequence 209~249 | 긴급 |

출력 행 수는 워밍업 29, 정상 93, 주의 49, 경고 38, 긴급 41이다. 모델이 네
서비스 단계를 실제로 출력했으며 데이터 파일의 원본 단계는 Backend 요청에
포함되지 않는다.

## Safety Fusion 해석

시간축 압축 때문에 원본 AI-Hub 2~3단계 구간의 고온 상승률이 커진다. 독립 물리
규칙은 sequence 122부터 긴급 조건을 충족하므로 최종 경보가 ML보다 먼저
긴급으로 올라갈 수 있다.

따라서 화면·발표에서 다음을 구분해야 한다.

- `ml_risk_level`: Hybrid HGB 4단계 시연 근거
- `physics_risk_level`: 독립 센서 안전 규칙
- `final_risk_level`: Safety Fusion 최종 경보

최종 경보가 네 단계를 순서대로 거친다고 주장하지 않는다. 보수적 물리 규칙이
ML보다 먼저 긴급을 낼 수 있다는 안전 구조를 함께 설명한다.

## 운영 주의사항

1. 전용 `CAR.car_id`와 해당 차량 접근 권한이 있는 일반 Spring JWT를 사용한다.
2. JWT는 `BMS_DEMO_JWT` 환경변수로만 전달하고 저장소·명령행 인수에 기록하지 않는다.
3. 시연 차량에 등록된 실제 `CHARGING_SESSION.session_id`, 0부터 증가하는
   sequence, 정확히 1초 증가하는 observedAt을 사용한다. 임의 UUID를 충전 세션으로
   사용하면 최초 주의 이상 저장 시 외래키 오류가 발생한다.
4. 주의 이상 매초 RDS 이상 로그와 보고서 작업이 만들어질 수 있다. 운영 시연 전
   사건 단위 중복 저장 정책을 결정하거나 별도 시연 환경을 사용한다.
5. 이 데이터는 96셀 실차 측정이 아니라 AI-Hub 실험 신호를 결정론적으로 확장한
   발표용 어댑터다. 실제 차량 성능 근거로 사용하지 않는다.
