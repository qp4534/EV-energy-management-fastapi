# 챗봇·AI 보고서 기능 가이드

AWS 기본 이미지는 하나의 `app.main:app` 프로세스에서 안전 추론, 챗봇, 보고서 Worker를 함께 실행한다.

- 통합 AWS API: `app.main:app`
- 사용자 챗봇·자동 보고서 Worker 단독 실행(선택): `app.chatbot.main:app`
- 월간·이상 보고서 작업 API: `app.reporting.main:app`
- 월간·이상 보고서 Worker 단독 실행(선택): `python -m app.reporting.worker`

기본 `Dockerfile`은 현재 GitHub Actions가 빌드하는 이미지이며 `EMBEDDED_AI_ENABLED=true`와 `REPORT_WORKER_ENABLED=true`를 설정한다. 따라서 AWS에서는 별도 Worker 명령 없이 챗봇과 보고서 Worker가 자동 시작된다. 사용량이 증가하면 `Dockerfile.ai`로 AI 기능만 다시 분리할 수 있다.

## 데이터 흐름

### 챗봇

```text
앱·웹
→ Spring 로그인·차량 권한 확인
→ POST /v1/chat/messages
→ 안전 우선 Supervisor
→ RAG / 최신 차량 상태 / 제한된 일반 답변
→ 출처가 포함된 응답
```

Supervisor는 LLM이 아니라 결정적인 코드 규칙이다.

- `EMERGENCY`: 화재·연기·폭발·감전 등. 승인된 안전자료만 검색하고 일반 LLM fallback을 금지한다.
- `VEHICLE_STATUS`: 기존 추론 API의 Redis 최신 스냅샷만 사용한다.
- `LEGAL`: 현행 공식 법령·고시 자료만 사용하고 검색 실패 시 추측하지 않는다.
- `RAG`: 일반 전기차·충전 질문을 문서에서 검색한다.
- `GENERAL`: 인사 등 낮은 위험도의 질문에만 일반 모델 답변을 허용한다.

현재 차량 상태는 다음 기존 추론 API에서 가져온다.

```http
GET /api/v1/twins/vehicles/{vehicle_id}/latest
```

이 API는 Redis의 `twin:latest:{vehicle_id}`만 읽는다. 과거 이상 프레임을 현재 상태로 대신 사용하지 않는다. 기본 30초보다 오래된 데이터는 챗봇에서 stale 상태로 처리한다.

### 보고서

```text
이상 이벤트 또는 월간 스케줄
→ ai_report_jobs 작업 생성
→ FastAPI 백그라운드 Worker가 작업 선점
→ 업무 DB에서 사실·수치 집계
→ RAG에서 안전 근거 검색
→ DeepSeek는 설명 문장만 생성
→ 고정 JSON 검증
→ AI_REPORTS 저장
```

DeepSeek가 실패해도 DB에서 계산한 지표를 이용한 기본 보고서는 저장할 수 있다. LLM은 집계값을 계산하거나 수정하지 않는다.

보고서에는 작성자, 검토자, 승인자, 서명란, 고객센터 연결 버튼을 생성하지 않는다.

## RAG 저장 구조

Alembic `20260806_0006` 마이그레이션은 다음을 만든다.

- PostgreSQL `vector` 확장
- `rag_documents`
- `rag_chunks`와 768차원 임베딩
- 벡터 HNSW 인덱스
- 키워드 GIN 인덱스
- `ai_report_jobs`

PDF·HWP·MD는 원문 보관 및 검수용이다. 실제 검색에는 검수된 JSONL을 적재한다.

JSONL 검증만 실행하면 DB와 임베딩 모델이 필요 없다.

```powershell
python -m app.rag.ingest "C:\path\to\docs\rag" --validate-only
```

실제 적재는 `--validate-only`를 제거한다.

```powershell
python -m app.rag.ingest "C:\path\to\docs\rag"
```

적재기는 다음을 검사한다.

- JSON 형식
- 필수 `chunk_id`, `content`
- 중복 `chunk_id`
- `content_sha256`
- 서로 다른 기존 JSONL 스키마 정규화
- 공식 자료와 내부 초안의 승인 상태 구분

`approved_for_deployment=false`인 내부 초안은 기본 검색에서 제외된다. 검토가 끝난 자료만 해당 값을 `true`로 변경해야 한다. 개발 중 초안 검색을 강제로 허용하는 설정은 존재하지만 운영 사용을 권장하지 않는다.

## 실행 프로세스

AWS 기본 통합 실행(안전 추론·챗봇·보고서 Worker):

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

`Dockerfile`에는 `EMBEDDED_AI_ENABLED=true`, `REPORT_JOBS_ENABLED=true`, `REPORT_WORKER_ENABLED=true`가 기본 설정되어 있다. 현재 GitHub Actions가 이 Dockerfile을 빌드하므로 코드를 배포하면 세 기능이 함께 시작된다.

배포 Workflow는 새 이미지로 `alembic upgrade head`를 먼저 실행한 뒤, 외부 GitOps Deployment에 통합 AI 환경변수, DeepSeek Secret 참조, 2Gi 요청/4Gi 메모리 상한과 새 이미지 태그를 함께 반영한다. 마이그레이션 실패 시 새 이미지는 배포하지 않는다.

챗봇·보고서 Worker만 단독 실행(선택):

```powershell
uvicorn app.chatbot.main:app --host 0.0.0.0 --port 8001 --workers 1
```

`Dockerfile.ai`에도 자동 Worker 설정이 있어 위 FastAPI가 시작되면 Worker가 함께 시작된다. 통합·단독 실행 모두 챗봇과 Worker가 같은 DB 연결 풀, RAG 검색기, 임베딩 모델, DeepSeek 클라이언트를 공유한다.

보고서 작업 API:

```powershell
uvicorn app.reporting.main:app --host 0.0.0.0 --port 8002 --workers 1
```

보고서 Worker 단독 실행(개발·장애 대응용 선택 사항):

```powershell
python -m app.reporting.worker
```

현재 운영 기본 이미지는 `Dockerfile`이며 안전 추론·챗봇·자동 보고서 Worker에 필요한 의존성을 모두 설치한다. `Dockerfile.ai`는 추후 AI 기능을 별도 서비스로 분리할 때 사용할 수 있다. 두 이미지 모두 별도 Worker 명령이 필요 없다.

## 런타임 설정 이름

저장소에는 실제 비밀값을 저장하지 않는다. 아래 이름은 로컬 프로세스 환경변수, GitHub Actions Secret 또는 AWS Secrets Manager/ECS Task Secret으로 주입한다.

| 이름 | 기본값/용도 |
|---|---|
| `AI_DATABASE_URL` | RAG와 보고서 Worker DB. 없으면 `DATABASE_URL` 사용 |
| `DEEPSEEK_API_KEY` | 서버 전용 DeepSeek 비밀키. 기본값 없음 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` |
| `DEEPSEEK_CHAT_MODEL` | `deepseek-v4-flash` |
| `DEEPSEEK_REPORT_MODEL` | `deepseek-v4-flash` |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` |
| `EMBEDDING_DIMENSION` | `768` |
| `INFERENCE_BASE_URL` | 기존 안전 추론 FastAPI 주소 |
| `AI_INTERNAL_TOKEN` | Spring과 내부 AI API 사이 선택적 토큰 |
| `REPORT_JOBS_ENABLED` | 이상 저장 트랜잭션에서 보고서 작업도 생성할지 여부 |
| `EMBEDDED_AI_ENABLED` | 기본 `app.main:app`에 챗봇·보고서 AI 런타임 포함 여부. `Dockerfile` 기본 `true` |
| `REPORT_WORKER_ENABLED` | FastAPI 안에서 보고서 Worker 자동 실행 여부. Docker 이미지 기본 `true` |
| `RAG_ALLOW_DRAFTS` | 기본 `false`; 운영에서는 그대로 유지 |

애플리케이션 코드는 `.env` 파일을 직접 만들거나 읽지 않는다. 런타임이 환경변수를 주입한다.

## Spring 연동 계약

### 챗봇 요청

```http
POST /v1/chat/messages
Content-Type: application/json
```

```json
{
  "userId": "사용자 식별자",
  "vehicleId": "차량 식별자",
  "message": "지금 내 차 배터리 상태는?",
  "conversationId": "선택값"
}
```

`userId`는 감사 로그용 식별자이며 권한 검사는 Spring이 먼저 수행해야 한다. FastAPI를 앱·웹에 직접 공개하지 않는다.

응답 예시:

```json
{
  "answer": "답변",
  "route": "VEHICLE_STATUS",
  "safetyLevel": "NORMAL",
  "dataAsOf": "2026-08-06T12:00:00Z",
  "sources": [],
  "missingFields": [],
  "fallbackUsed": false,
  "metadata": {}
}
```

### 이상 보고서 작업

```http
POST /internal/v1/report-jobs/anomalies/{anomalyId}
```

### 월간 보고서 작업

```http
POST /internal/v1/report-jobs/monthly
```

```json
{
  "carId": "차량 UUID",
  "targetMonth": "2026-07"
}
```

두 API는 중복 요청을 받아도 동일한 `job_key`를 사용한다.

- `ANOMALY:{anomaly_id}`
- `MONTHLY:{car_id}:{YYYY-MM}`

위 영문 값은 작업 큐 내부 식별자다. 실제 운영 DB의
`AI_REPORTS.report_type`과 보고서 JSON의 `reportType`에는 프론트 필터와
동일하게 `월간보고서` 또는 `이상`을 저장한다.

Worker는 `FOR UPDATE SKIP LOCKED`로 작업을 선점하며 실패 시 최대 횟수까지 재시도한다. 저장되는 `report_id`도 작업 키에서 결정적으로 생성해 작업 재실행으로 보고서가 중복되지 않게 한다.

## 현재 DB에서 계산하는 보고서 항목

이상 보고서:

- 이상 유형, 감지 시각, 위험등급
- 당시 `TWIN_FRAMES` 위험 단계
- `model_input`에 실제 존재하는 온도·전압 값
- RAG 근거가 있는 안전 조치

월간 보고서:

- 충전 세션 수와 완료 세션 수
- 종료 시각이 있는 세션의 총 충전 시간
- 시작·종료 SOC가 모두 있는 세션의 평균 SOC 변화
- 위험등급별 이상 발생 수
- 저장된 이상 프레임의 월간 온도 통계

현재 DB에 충전 전력량 또는 주행거리 원천 데이터가 없으므로 총 충전 kWh, 전비, 주행거리 등은 생성하지 않는다. 값이 없으면 `missingFields`로 표시한다.

## 테스트

```powershell
python -m pytest -q
```

테스트는 실제 DeepSeek, Redis, PostgreSQL을 호출하지 않고 가짜 구현을 사용한다. 다음을 검증한다.

- 긴급 질문에서 일반 LLM fallback 금지
- 법령 근거가 없을 때 조항 생성 금지
- 오래된 차량 상태 거부
- RAG 출처 응답
- DeepSeek 요청 모델과 JSON 모드
- 보고서 숫자 보존
- DeepSeek 장애 시 기본 보고서 생성
- JSONL 해시와 중복 검증
