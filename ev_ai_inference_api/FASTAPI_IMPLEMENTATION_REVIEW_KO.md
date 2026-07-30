# FastAPI 구현 검토

## 생성 구조와 책임

`app/main.py`는 lifespan·router 등록만 담당한다. `routers/current_stage.py`는 HTTP 파싱/상태코드, `services/current_stage_service.py`는 기존 `HybridSafetySupervisorV2` 호출과 결과 변환, `core/session_manager.py`는 차량별 Supervisor·Lock·TTL·최대 세션을 담당한다. `schemas/current_stage.py`는 Pydantic 요청/응답 계약이다. 원본 배포 번들은 `model_bundles/current_stage_v1/ev_battery_safety_inference_v1/`에 복사했으며, joblib과 학습/4단계 매핑은 변경하지 않았다.

## 모델·입출력 계약

lifespan은 manifest의 SHA256, 두 joblib, calibration, policy를 검증하고 Supervisor를 생성해 로딩과 feature 계약을 확인한다. 요청마다 모델 파일을 읽지 않는다. 0~29개는 `warming_up`, 30~119개는 `stage_30s`, 120개부터 `stage_120s`다. 확률은 calibration 후 현재 4단계 분류 확률이며 합계는 1이고 미래 화재/180초 위험 확률이 아니다.

필수 입력 누락·JSON 구조 오류·NaN/Infinity는 422다. 물리 범위 밖 필수 신호는 Safety Supervisor가 받아 `invalid/unknown`으로 반환하며 이력을 초기화한다. 중복/역순 timestamp는 409, 1초 단절은 명시적으로 세션을 reset한 뒤 새 표본부터 시작한다. API에는 stack trace를 노출하지 않는다.

## Safety Fusion

ML pattern, 물리 규칙, 최종 경보를 별도 필드로 반환한다. ML emergency만으로 final emergency가 되지 않는다. 온도 단일 근거도 차단되며, 정책의 독립 온도 또는 열-전기 교차 근거가 있을 때만 hard physical emergency가 가능하다. 충전건 온도는 `charging_equipment_observation`에만 반영된다.

## 배포 구조와 한계

Docker는 Python 3.11 slim, non-root 사용자, 포트 8000, worker 1개 및 번들 포함으로 구성했다. Kubernetes는 replica 1, `/healthz` liveness, `/readyz` readiness 및 보수적 requests/limits를 제공한다. 메모리 세션 MVP이므로 복수 worker/replica에는 안전하지 않다. 생산 확장은 Redis/ElastiCache 또는 Spring Boot가 차량 이력을 소유하도록 SessionManager를 교체해야 한다.

CI/CD는 의도적으로 GitHub Actions/CodePipeline 중립이다. 어느 선택이든 테스트 → `verify_bundle.py` → Docker build → ECR push → EKS update만 외부 pipeline에 배치한다. AWS 리소스 생성·push·배포는 수행하지 않았다.

## 검증 기록

테스트에는 health/ready/model-info, 29/30/120 routing, timestamp reset·중복 거절, 필수값/NaN, 범위 invalid, 차량 A/B 격리, 동일 차량 동시 요청 Lock 직렬화, reset/delete, 확률 합, 온도 단일 emergency 차단, 교차근거 emergency, 충전건 온도 격리가 포함된다. `tf_env`에서 번들 `verify_bundle.py`는 PASS, `pytest -q`는 **6 passed (33.48s)**였다.

Docker build는 두 번 시도했다. 첫 시도는 sandbox의 Docker 설정 접근 제한으로 실패했고, 승인된 재시도는 Docker Desktop Linux daemon named pipe가 존재하지 않아 실패했다. Dockerfile 자체의 이미지 빌드·실행은 daemon을 시작한 뒤 `docker build -t ev-ai-inference-api .`로 재검증해야 한다. 이 작업에서는 이미지 push나 AWS 배포를 수행하지 않았다.
