# AI 서비스 (dev_nh)

메인 백엔드와 별도로 띄우는 두 개의 독립 FastAPI 모델 서빙 서비스.

| 서비스 | 포트 | 역할 | ERD 연동 대상 |
|---|---|---|---|
| [`charging-demand/`](charging-demand/) | 8001 | 시간대별 충전 수요 예측 | `CHARGER.queue_length`, `CHARGER.waiting_time_min` |
| [`rul-diagnosis/`](rul-diagnosis/) | 8000 | 배터리 화재위험/SOH등급/잔여수명·매각가 3단계 진단 | `BATTERY_PASSPORT`, `BATTERY_DIAGNOSIS_METRICS`, `BATTERY_PROPOSALS`/`BATTERY_OFFERS`(가격 산정 참고용) |

각 폴더의 `README.md`에 실행 방법, 모델 파일 준비(대용량 2개는 git 미포함), ERD 컬럼명에 맞춘 어댑터 엔드포인트(`/diagnose/erd`) 설명이 있다.

## 왜 두 서비스를 메인 앱에 합치지 않았나
- 모델 로딩(특히 `rul-diagnosis`)이 무겁고(수백MB), 배포 사이클이 다르다(모델 재학습 ≠ 백엔드 배포).
- 두 서비스 다 이미 그 자체로 완결된 FastAPI 앱(`fastapi_app.py`)이라, 그대로 독립 프로세스/컨테이너로 띄우고 메인 백엔드는 HTTP로만 호출하는 편이 배포·확장에 유리하다고 판단했다.
- 대용량 모델 파일(304MB, 125MB)이 GitHub 100MB 제한을 넘어서, 이걸 우회하려고 Git LFS를 도입하는 대신 "모델은 로컬/별도 서버에 두고 서비스만 띄운다" 방식을 택했다. (자세한 배경은 각 서비스 README 참고)
