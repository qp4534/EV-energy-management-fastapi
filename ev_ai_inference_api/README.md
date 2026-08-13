# EV Battery Safety Inference API

현재 BMS형 4단계 상태 분류 모델을 FastAPI로 제공하는 MVP입니다. 이 서비스는 향후 180초 위험진입 확률, OEM BMS, 실제 차량 안전 인증 모델이 아닙니다.

## 실행

지정 환경에서 `C:\Users\User\anaconda3\envs\tf_env\python.exe -m pip install -r requirements.txt` 후 다음을 실행합니다.

```powershell
C:\Users\User\anaconda3\envs\tf_env\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

`MODEL_BUNDLE_DIR`로 번들 루트를 바꿀 수 있습니다. 시작 시 joblib 모델 두 개, manifest, calibration, 정책 및 SHA256을 한 번만 검증·로드합니다. `/healthz`는 프로세스 생존, `/readyz`는 이 로딩 검증 완료를 뜻합니다.

필수 1Hz 입력은 `voltage_v`(pack 전압이 아닌 대표 셀 전압, 0~6V), `temp_mean_c`, `temp_max_c`, `temp_delta_c`, `temp_saturation_fraction`, `temp_saturation_all`입니다. 이전 값 보간은 하지 않습니다. 범위 밖 값은 모델 이력에 추가되지 않고 `sensor_health=invalid`, `final_safety_alert=unknown`을 반환합니다. `charging_gun_temperature_c`는 충전 설비 관찰값일 뿐 셀 온도로 사용하지 않습니다.

각 차량은 독립 Supervisor·Lock·최근 시각을 갖습니다. 시각이 1초 연속이 아니면 이력을 초기화하고, 중복 또는 과거 시각은 HTTP 409으로 거절합니다. 세션은 TTL(기본 900초) 후 정리되고 최대 1000개로 제한됩니다. reset/delete는 각각 이력 초기화와 완전 제거입니다.

이 MVP는 프로세스 메모리에만 이력을 보관하므로 **Uvicorn worker 1개, Kubernetes replica 1개**로 운용해야 합니다. 다중 worker/Pod에서는 차량 이력이 분산됩니다. sticky session은 완전한 해결책이 아닙니다. 이후 Spring Boot 또는 Redis/ElastiCache가 차량별 120초 이력을 소유하도록 `SessionManager` 책임을 대체할 수 있습니다.

## Docker와 Kubernetes

```powershell
docker build -t ev-ai-inference-api .
docker run -p 8000:8000 ev-ai-inference-api
```

### 로컬 3D Twin 실험

Docker Desktop이 실행 중일 때 프로젝트 전용 PostgreSQL 17은 Windows의 기존
PostgreSQL 5432와 분리해 `127.0.0.1:5433`에 노출됩니다. Redis와 API도 각각
`127.0.0.1:6379`, `127.0.0.1:8000`으로만 노출됩니다.

```powershell
docker compose up --build -d
docker compose ps
docker compose exec api python -m app.simulator seed-history
```

`seed-history`와 `replay-live`는 기본적으로 RDS `CAR` 테이블에서 차량번호 순으로
실제 `car_id` UUID 10개를 읽어 시연 프로필에 결합합니다. 다른 차량 집합을 고정하려면
`TWIN_DEMO_VEHICLE_IDS` 환경변수 또는 `--vehicle-ids` 옵션에 쉼표로 구분한 UUID 10개를
지정합니다. UUID가 아닌 `car-uuid-001` 같은 mock 식별자는 실시간 시연에 사용하지 않습니다.

`seed-history`는 위험 프로필이 배정된 실제 차량에 완료된 3시간 사건과 차량당 정확히
10,800개 프레임을 만듭니다. 실시간 전송은
Windows 호스트의 가상환경에서 실행하며 `--speed`는 논리적 1Hz 대비 배속입니다.

```powershell
.\.venv\Scripts\python.exe -m app.simulator replay-live --speed 60
```

Twin API는 다음 계약을 제공합니다.

- `POST /api/v1/twins/vehicles/{vehicle_id}/samples`
- `GET /api/v1/twins/vehicles/{vehicle_id}/latest`
- `GET /api/v1/twins/vehicles/{vehicle_id}/latest/measurement?stale_after_seconds=10`
- `GET /api/v1/twins/risk-vehicles`
- `GET /api/v1/twins/vehicles/{vehicle_id}/incidents`
- `GET /api/v1/twins/vehicles/{vehicle_id}/incidents/latest/history?resolution_seconds=30`
- `GET /api/v1/twins/vehicles/{vehicle_id}/incidents/{incident_id}/history?resolution_seconds=30`
- `WS /api/v1/twins/vehicles/{vehicle_id}/live`

공개 `TwinFrame`은 `generic_ev_concept_96_v1` 배치의 96셀 온도·전압·상태 배열과
3개 커넥터 부품 상태를 담습니다. 사건은 주의 이상 2초 연속 시 즉시 시작하며,
아직 한 시간 버퍼가 차지 않았으면 가용한 연속 사전 프레임부터 저장합니다.
`[발생-3600초, 발생+7200초)`에 정확히 10,800개가 모이면 `complete`, 창이 끝났지만
프레임이 부족하면 `incomplete`로 닫습니다. 60초 정상 상태로 재무장되면 이전 사건의
사후 창이 진행 중이어도 새 사건을 시작하며, 겹치는 프레임은 두 사건 모두에 저장합니다.
`final_risk_level`은 기존 `HybridSafetySupervisorV2`의 `final_safety_alert`를 그대로
사용합니다. 셀·커넥터 임계값과 열화상 결과는 3D 시각화용 상태이며 이 최종 판정을
덮어쓰지 않습니다. `fusion_source`는 셀·모듈 시각화에 사용된 입력을 설명합니다.
기존 `/v1/vehicles/{vehicle_id}/samples` 계약은 그대로 유지됩니다.

### 디지털 트윈 셀 위험 분석 AI

`cell-risk-graphsage-v1`은 96개 셀을 6x16 인접 그래프로 구성하고 현재 셀 온도·전압,
BMS 위험 단계, 열화상 AI의 위험 단계·신뢰도·핫스팟을 결합해 셀별 위험 단계와 핵심
위험 셀, 영향 셀 범위, 확산 방향을 분류합니다. 미래 온도를 예측하지 않으며 차량의
`final_risk_level`도 변경하지 않습니다. 실시간 차량 입력, S3 시나리오 재생, PostgreSQL
사고 이력 조회에서 같은 모델을 사용합니다. 운영 추론은 작은 NumPy 번들로 수행해
재생 워커가 PyTorch를 메모리에 올리지 않습니다.

모델은 생성된 시나리오 데이터로 학습한 3D 시각화용 프로토타입입니다. 번들의 내부
검증 수치를 실차 안전 성능이나 외부 데이터 일반화 성능으로 해석하면 안 됩니다.

배터리 여권처럼 현재 상태를 표시하는 화면은 `latest/measurement`의
`max_cell_temperature_c`를 "현재 최고 셀 온도"로 표시하고 `observed_at`,
`is_stale`을 함께 확인해야 합니다. 이 값은 Redis의 차량별 최신 Twin 프레임에서만
계산하며 `BATTERY_PASSPORT.current_temp`와 섞지 않습니다. 반면 이상 안전보고서는
해당 `anomaly_id`에 연결되고 감지 시각에 가장 가까운 `TWIN_FRAMES.model_input`만
사용합니다. 사건 프레임에 온도가 없으면 여권 온도로 대체하지 않고 측정 데이터 없음으로
표시해 과거 사건과 현재 상태가 섞이지 않도록 합니다.

`k8s/`는 일반 Kubernetes manifest만 제공하며 이미지 URI는 `REPLACE_WITH_IMAGE_URI` placeholder입니다. AWS 계정·리전·ECR·EKS 정보는 포함하지 않았습니다.

## CI/CD 선택 중립성

GitHub Actions를 선택하면 `.github/workflows/deploy.yml`에서 테스트 → `verify_bundle.py` → Docker build → ECR push → EKS update를 둡니다. CodePipeline을 선택하면 `buildspec.yml`/CodeBuild에서 동일한 순서를 둡니다. 이 저장소의 FastAPI 코드·Dockerfile·manifest는 어느 쪽도 전제하지 않으며, 실제 파이프라인과 AWS 리소스는 만들지 않습니다.

향후 모델은 `routers/onset_180s.py`, `services/onset_180s_service.py`, `schemas/onset_180s.py`, `model_bundles/onset_180s_v1/`처럼 독립 수직 슬라이스로 추가합니다. thermal vision·SOH/RUL도 같은 방식이며, 현 작업에는 placeholder를 만들지 않았습니다.

## Hybrid HGB 실제 추론 시연

발표용 BMS 센서값을 Spring Backend 경유로 전송해 배포된 Hybrid HGB와
Safety Fusion을 실제로 실행하는 절차는
[`demo/bms_hgb/README_KO.md`](demo/bms_hgb/README_KO.md)에 정리되어 있습니다.
이 시연은 RDS에 저장된 위험등급을 재생하지 않으며, 입력에 원본 단계 라벨을
포함하지 않습니다.
