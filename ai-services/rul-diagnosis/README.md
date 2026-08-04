# 배터리 진단 멀티에이전트 서비스 (rul-diagnosis)

7개 센서값(방전/충전 특성)과 8개 화재 센서값을 입력받아 3단계 에이전트(화재 위험 게이트 → SOH 등급 → 잔여수명/매각가)를 실행하는 독립 FastAPI 서비스.

## 모델 파일 준비 (중요)
이 디렉터리에는 `reuse_model.joblib`(14MB)만 포함되어 있다. 아래 두 파일은 **GitHub 100MB 제한을 넘어 저장소에 커밋하지 않았다** (`.gitignore` 처리됨):

- `rul_model_B_no_cycle.joblib` (약 304MB)
- `fire_risk_model.joblib` (약 125MB)

서버를 띄우기 전에 이 두 파일을 **이 디렉터리에 직접 복사**해 두어야 한다(원본: `C:\Users\User\Desktop\빅프\rul_diagnosis\`). 파일이 없으면 서버 기동 시 어떤 파일이 없는지 알려주는 에러 메시지가 뜬다.

장기적으로는 아래 중 하나로 정리하는 걸 권장:
- 팀 공유 스토리지(S3 등)에 올려두고 배포 스크립트에서 받아오기
- 또는 이 서비스를 별도로 계속 띄워두고(로컬/사내 서버), 메인 백엔드는 URL로만 호출(아래 "다른 서비스에서 호출하기" 참고) — 모델을 아예 git에 안 올려도 됨

## 실행
```bash
pip install -r requirements.txt
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```
`ANTHROPIC_API_KEY` 환경변수는 `/pipeline`에서 `include_report=true`일 때만 필요(종합 리포트 생성용). `/diagnose/erd`, `/agents/*`, `include_report=false`인 `/pipeline`은 키 없이 동작한다.

## 다른 서비스에서 호출하기 (메인 백엔드 연동)
모델 바이너리를 매 배포마다 옮기지 않으려면, 이 서비스를 독립적으로 띄워두고 메인 백엔드는 URL만 알면 된다.
- 로컬 개발: `http://localhost:8000`
- 임시 외부 연동 테스트(같은 팀원 PC에서 띄운 걸 다른 곳에서 호출해봐야 할 때): `ngrok http 8000`으로 임시 공개 URL을 받아서 사용. ngrok URL은 세션마다 바뀌고, 그 사이 이 API가 인터넷에 노출되니 **개발/테스트 용도로만** 짧게 쓰고 끝나면 tunnel을 끄는 걸 권장.
- 운영 배포: 별도 컨테이너로 배포 후 내부 서비스 URL(`RUL_DIAGNOSIS_API_URL` 같은 환경변수)로 관리.

## 엔드포인트
- `GET /health`
- `POST /agents/safety-guard` / `/agents/status-classifier` / `/agents/value-assessor` — 에이전트 단독 호출
- `POST /pipeline` — 3단계 게이트 파이프라인 (+선택적 Claude 종합 리포트)
- `POST /pipeline/batch-excel` — 엑셀 일괄 진단
- **`POST /diagnose/erd`** — 위 파이프라인을 실행하고 **ERD 컬럼명**(`battery_level`, `reuse_status`, `grade_detail`, `reliability_score`, `rul`, `remaining_life_score`, `discharge_power_score`, `charge_health_score`, `voltage_stability_score`)으로 매핑해서 반환. `BATTERY_PASSPORT`/`BATTERY_DIAGNOSIS_METRICS` 테이블에 바로 적재하기 위한 용도. 응답 안에 스케일/매핑 관련 주의사항이 같이 내려가니 백엔드팀과 한 번 맞춰볼 것.
