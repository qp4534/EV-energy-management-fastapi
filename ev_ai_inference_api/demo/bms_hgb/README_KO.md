# Hybrid HGB 실제 추론 시연

이 시연은 RDS 위험등급을 화면에 복사하지 않는다. AI-Hub 보류 실험의 전압과
두 온도 신호를 96셀 BMS 입력 계약으로 변환한 뒤 Spring Backend에 1초 논리
간격으로 전송한다.

```text
JSONL 센서값
  -> Spring Backend /api/twin-frames/cars/{carId}/bms-samples
  -> FastAPI Twin pipeline
  -> Hybrid HGB + 물리 규칙 + Safety Fusion
  -> Redis 실시간 화면
  -> 주의 이상이면 RDS ANOMALY_LOGS/TWIN_FRAMES.model_input
```

## 데이터의 의미

- 원본: AI-Hub 통제 열폭주 실험 `20250912005`
- 분할: 기존 잠금 test 실험
- 매핑: AI-Hub 1=정상, 2~3=주의, 4~5=경고, 6=긴급
- 시연 데이터: 정상 120초, 주의 50초, 경고 50초, 긴급 30초
- 96셀 온도: 표면온도 48셀 + 양극단자온도 48셀
- 96셀 전압: AI-Hub 전압을 96셀에 동일하게 복제
- 커넥터 온도: 임의 고장을 만들지 않고 주변 온도를 사용

서비스 단계는 생성 과정을 검토하기 위한 메타데이터로 JSONL에 남지만 Backend
요청에는 절대 포함되지 않는다. `ml_risk_level`은 배포된 Hybrid HGB가 새로
계산한다.

이 자료는 발표 시간에 맞춘 시계열 압축·96셀 어댑터 데이터다. 실제 차량 96셀
측정 데이터나 새로운 성능 검증셋으로 주장하면 안 된다.

## 1. 데이터 생성

프로젝트 루트 `ev_ai_inference_api`에서 실행한다.

```powershell
python -m app.bms_demo generate `
  --source-csv "C:\Users\User\aihub_thermal_runaway_tl_clean.csv" `
  --experiment-id 20250912005 `
  --output demo\bms_hgb\aihub_holdout_20250912005_demo.jsonl `
  --manifest demo\bms_hgb\manifest.json
```

저장소에는 동일 명령으로 생성하고 검증한 JSONL과 manifest를 포함한다.

## 2. JWT·전용 차량·충전 세션 준비

Spring Backend에 로그인한 사용자가 접근 가능한 `CAR.car_id` UUID를 전용 시연
차량으로 선택한다. 일반 사용자 JWT를 코드나 Git에 저장하지 않고 현재
PowerShell 세션 환경변수에만 넣는다.

```powershell
$env:BMS_DEMO_JWT = "로그인 후 받은 JWT"
```

시연 차량은 실제 `CHARGING_SESSION` 한 건과 연결되어야 한다. Backend의
`POST /api/charging-sessions`로 세션을 만들고 응답의 `sessionId`를 보관한다.
`chargerId`에는 RDS에 이미 존재하는 시연 충전기의 UUID를 사용한다.

```powershell
$sessionBody = @{
  startTime  = (Get-Date).ToUniversalTime().ToString("o")
  endTime    = $null
  changeState = "충전중"
  carId      = $carId
  chargerId  = $chargerId
  startSoc   = 60.0
  endSoc     = $null
} | ConvertTo-Json

$demoSession = Invoke-RestMethod `
  -Method Post `
  -Uri "${backend}/api/charging-sessions" `
  -Headers $headers `
  -ContentType "application/json; charset=utf-8" `
  -Body $sessionBody

$chargingSessionId = $demoSession.sessionId
```

## 3. Backend 경유 재생

최종 발표는 `--speed 1`이 실제 1Hz와 일치한다. 리허설만 5배속 이하를 권장한다.

```powershell
python -m app.bms_demo replay `
  --dataset demo\bms_hgb\aihub_holdout_20250912005_demo.jsonl `
  --backend-url "https://백엔드주소" `
  --car-id "전용-시연-CAR-UUID" `
  --charging-session-id "전용-시연-CHARGING_SESSION-UUID" `
  --speed 1 `
  --result runtime\bms_hgb_demo_result.jsonl
```

모델은 처음 29개 샘플까지 `warming_up`, 30번째부터 30초 HGB, 120번째부터
120초 장기 추세 HGB를 사용한다. `sessionId`에는 Backend가 생성한 실제
`CHARGING_SESSION.session_id`를 사용한다. 따라서 주의 이상 결과가 RDS에 저장될
때도 외래키 제약을 위반하지 않는다.

발표 시간을 줄이려면 AI 시연 순서 약 2분 전에 정상 120초 구간을 백그라운드로
시작한다. 발표자가 화면을 설명할 때 주의·경고·긴급 구간이 이어진다.

## 4. 결과 확인

콘솔과 결과 JSONL에서 다음을 구분한다.

- `ml_risk_level`: Hybrid HGB 실제 판정
- `physics_risk_level`: 독립 물리 규칙
- `final_risk_level`: Safety Fusion 최종 결과
- `anomaly_id`: 주의 이상 RDS 저장 결과

주의 이상 매초가 현재 RDS 이상 로그로 저장될 수 있으므로 반드시 전용 시연
차량을 사용한다. 보고서 목록을 오염시키지 않으려면 운영 시연 전 사건 단위
중복 저장 정책을 별도로 결정해야 한다.
