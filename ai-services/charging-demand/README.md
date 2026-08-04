# 충전 수요 예측 서비스 (charging-demand)

시간대·요일·월·연도를 입력받아 시간당 예상 충전 세션 수를 예측하는 독립 FastAPI 서비스.
모델(`demand_ca_model.joblib`, 532KB)은 저장소에 그대로 포함되어 있음(용량 문제 없음).

## 실행
```bash
pip install -r requirements.txt
uvicorn fastapi_app:app --host 0.0.0.0 --port 8001
```

## 엔드포인트
- `GET /health`
- `POST /predict` — `{hour, dow, month, year}` → 예상 세션 수 / 수요 수준
- `POST /predict/curve` — 하루 0~23시 곡선

## ERD 연동 메모
- `CHARGER.queue_length`, `CHARGER.waiting_time_min`은 ERD 주석상 "AI 모델 출력값, 주기 배치 갱신" 대상 컬럼이다.
  이 서비스의 `/predict` 결과(예상 세션 수·수요 수준)를 `BATCH_JOBS`에 등록된 배치가 주기적으로 호출해
  각 `CHARGING_STATION`/`CHARGER`의 대기열·예상 대기시간으로 환산해 UPDATE하는 흐름을 권장한다.
  구체적인 세션수→대기열 환산식은 아직 확정되지 않았으므로 백엔드팀과 협의해서 정하면 됨.
- 재학습은 `train_demand_ca.py` 참고(학습 데이터는 이 저장소에 포함하지 않음).
