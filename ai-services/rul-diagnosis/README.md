# 배터리 진단 멀티에이전트 서비스 (rul-diagnosis)

7개 센서값(방전/충전 특성)과 8개 화재 센서값을 입력받아 3단계 에이전트(화재 위험 게이트 → SOH 등급 → 잔여수명/매각가)를 실행하는 독립 FastAPI 서비스.

## `fire_risk_model`과 `ev_ai_inference_api`의 차이
둘 다 열폭주 위험을 예측하지만 같은 걸 두 번 만든 게 아니다. 요구하는 입력 자체가 겹치지 않아서 하나로 합칠 수 없다.

| | `fire_risk_model` | `ev_ai_inference_api` |
|---|---|---|
| 목적 | 매도 제안서 파이프라인의 1회성 안전 게이트 — 진단을 계속 진행해도 되는지만 판정 | 주행 중인 실제 차량의 실시간 관제 대시보드 |
| 입력 | 전압·온도·압력·열화상온도·가스 8종(CH4/CO/CO2/HCN/HCl/HF/N2O/NO/NO2/SO2) 등 30개 — 실험실/진단 장비로만 계측 가능 | `voltage_v`, `temp_mean_c`, `temp_max_c`, `temp_delta_c`, `temp_saturation_fraction/all` — 양산차 BMS가 실시간으로 주는 값만 |
| 호출 방식 | 센서값 스냅샷 1개(dict) → 1회 판정, 상태 없음 | 차량별 세션 유지, 1Hz 연속 스트림, TTL 900초 |
| 모델 | RandomForestClassifier(n_estimators=300) 2개 — 6단계(`tr_stage` 1~6) 분류 + 이진(stage≥4) 위험판정 | Hybrid HistGradientBoosting — 4단계(정상/주의/경고/긴급) 분류 |
| 학습 데이터 | AIHub 열폭주 실험 데이터, `experiment_id` 단위 GroupShuffleSplit(75/25)로 정보 누설 차단 | `ev_ai_inference_api/README.md` 참고 |
| 성능 (테스트 56,396행) | 6단계 정확도 92.4%/macro-F1 85.8%, 이진판정 ROC-AUC 0.992/Recall 89.1% (운영 임계값은 70%로 보수적 설정) | 긴급 단계 Recall 약 96% |

주행 중인 차량에서는 가스 농도나 열화상을 실시간으로 뽑을 방법이 없어서 `ev_ai_inference_api`를 이 파이프라인에 그대로 재사용할 수 없다. 반대로 `fire_risk_model`이 요구하는 실험실 수준 센서값은 양산차 BMS에서 얻을 수 없다.

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
`DEEPSEEK_API_KEY_NH` 환경변수는 `/pipeline`에서 `include_report=true`일 때만 필요(종합 리포트 생성용). `/diagnose/erd`, `/agents/*`, `include_report=false`인 `/pipeline`은 키 없이 동작한다.

## 다른 서비스에서 호출하기 (메인 백엔드 연동)
모델 바이너리를 매 배포마다 옮기지 않으려면, 이 서비스를 독립적으로 띄워두고 메인 백엔드는 URL만 알면 된다.
- 로컬 개발: `http://localhost:8000`
- 임시 외부 연동 테스트(같은 팀원 PC에서 띄운 걸 다른 곳에서 호출해봐야 할 때): `ngrok http 8000`으로 임시 공개 URL을 받아서 사용. ngrok URL은 세션마다 바뀌고, 그 사이 이 API가 인터넷에 노출되니 **개발/테스트 용도로만** 짧게 쓰고 끝나면 tunnel을 끄는 걸 권장.
- 운영 배포: 별도 컨테이너로 배포 후 내부 서비스 URL(`RUL_DIAGNOSIS_API_URL` 같은 환경변수)로 관리.

## 엔드포인트
- `GET /health`
- `POST /agents/safety-guard` / `/agents/status-classifier` / `/agents/value-assessor` — 에이전트 단독 호출
- `POST /pipeline` — 3단계 게이트 파이프라인 (+선택적 DeepSeek 종합 리포트)
- `POST /pipeline/batch-excel` — 엑셀 일괄 진단
- **`POST /diagnose/erd`** — 위 파이프라인을 실행하고 **ERD 컬럼명**(`battery_level`, `reuse_status`, `grade_detail`, `reliability_score`, `rul`, `remaining_life_score`, `discharge_power_score`, `charge_health_score`, `voltage_stability_score`)으로 매핑해서 반환. `BATTERY_PASSPORT`/`BATTERY_DIAGNOSIS_METRICS` 테이블에 바로 적재하기 위한 용도. 응답 안에 스케일/매핑 관련 주의사항이 같이 내려가니 백엔드팀과 한 번 맞춰볼 것.
- **`POST /report/pdf`** — 배터리 매도 제안서 PDF를 생성해 그대로 스트리밍 반환(`application/pdf`). Agent1~3을 실행한 뒤 `valuation.estimate_offers()`로 매입처를 매칭하고(`buyer_index`로 순위 선택, 기본 0 = 최고 제안가), `economics.compute()`로 경제성·탄소 절감을 계산해 `pdf_report.build_pdf()`로 렌더링한다. 화재 위험 게이트에 걸리면 제안서를 만들지 않고 409를 반환한다.
  - 한글 폰트는 `fonts/NotoSansKR-Regular.ttf` / `NotoSansKR-Bold.ttf`(OFL 라이선스, `fonts/OFL.txt` 참고)를 이 저장소에 함께 담아 배포한다 — 배포 컨테이너(Linux)에는 시스템 한글 폰트가 없어서, 로컬 윈도우 전용 "맑은 고딕" 대신 리포에 번들된 폰트를 우선 쓰도록 `pdf_report.py`에서 처리했다.
