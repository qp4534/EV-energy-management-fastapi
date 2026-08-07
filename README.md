# EV-energy-management-fastapi

## BMS 현재 상태 분류 API

백엔드는 차량의 BMS 수치 JSON을 차량 ID와 함께 FastAPI로 전송합니다.

```text
POST /v1/vehicles/{car_id}/samples
```

`car_id`는 RDS `CAR.car_id`의 UUID를 사용합니다.

```json
{
  "timestamp_seconds": 30,
  "voltage_v": 3.92,
  "temp_mean_c": 39.4,
  "temp_max_c": 43.1,
  "temp_delta_c": 3.7,
  "temp_saturation_fraction": 0.0,
  "temp_saturation_all": false,
  "observed_at": "2026-08-05T10:30:00+09:00"
}
```

핵심 BMS 수치 6개는 필수이며, `observed_at`, 열화상 관련 값은 선택값입니다.
30초 이상의 순차 샘플이 쌓이면 모델 분류 결과가 반환됩니다.

## 이상 탐지 시 RDS 저장

`ANOMALY_PERSISTENCE_ENABLED=true`로 실행하면 `caution`, `warning`,
`emergency` 결과만 RDS에 저장합니다. `normal`과 `unknown` 결과는 저장하지
않습니다.

1. `ANOMALY_LOGS`에 이상 로그를 생성합니다.
2. 생성된 `anomaly_id`를 사용해 `TWIN_FRAMES`를 생성합니다.
3. 원본 요청과 추론 결과는 `TWIN_FRAMES.raw_metrics` JSONB에 저장합니다.
4. API 응답의 `anomaly_id`로 백엔드가 저장된 이상 건을 조회할 수 있습니다.

RDS의 위험도 컬럼은 `VARCHAR + CHECK` 구조이며 FastAPI는 모델 단계 값을
`정상`, `주의`, `경고`, `긴급`으로 변환해 저장합니다.

### 환경변수

```env
DATABASE_URL=postgresql+asyncpg://<user>:<password>@<rds-host>:5432/<database>
ANOMALY_PERSISTENCE_ENABLED=true
```

DB 접속 정보는 Git에 넣지 말고 Kubernetes Secret 또는 배포 환경의 Secret으로
주입합니다.

## 검증

```text
pytest -q
```

현재 테스트는 모델 분류, 이상 결과만 저장 대상으로 전달되는 계약, 입력 검증을
검사합니다.

## AWS 기본 실행 구조

기본 `Dockerfile`과 `app.main:app` 하나가 다음 기능을 함께 실행합니다.

- 기존 BMS 안전 추론 API
- 사용자 챗봇 `POST /v1/chat/messages`
- 월간·이상 보고서 백그라운드 Worker

Docker 이미지에는 `EMBEDDED_AI_ENABLED=true`, `REPORT_JOBS_ENABLED=true`,
`REPORT_WORKER_ENABLED=true`가 기본 설정되어 있습니다. 따라서 GitHub Actions가
기본 이미지를 배포하면 별도 Worker 명령 없이 함께 시작됩니다. 운영 비밀값인
`DEEPSEEK_API_KEY`와 DB 접속 정보는 GitHub에 넣지 않고 Kubernetes Secret으로
주입합니다.

배포 Workflow는 새 이미지를 이용해 `alembic upgrade head`를 먼저 실행합니다.
마이그레이션이 성공한 경우에만 GitOps 이미지 태그를 갱신하므로 RAG·보고서 작업
테이블이 준비되지 않은 이미지가 서비스에 먼저 배포되지 않습니다.

GitOps의 FastAPI Deployment도 `fastapi-secret`에서 `DEEPSEEK_API_KEY`를 읽고,
통합 이미지에 최소 2GiB 요청/4GiB 상한을 주도록
맞춰야 합니다. 이 저장소의 `ev_ai_inference_api/k8s/deployment.yaml`이 기준 예시입니다.
