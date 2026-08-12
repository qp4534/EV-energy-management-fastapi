# Cell Risk GraphSAGE v1

96개 배터리 셀의 현재 온도·전압과 BMS/열화상 AI 결과를 입력으로 받아
셀별 위험 단계, 핵심 위험 셀, 영향 셀 범위와 열 확산 방향을 분류하는
디지털 트윈 시각화 전용 모델입니다.

- 구조: 2-layer GraphSAGE node classifier
- 셀 그래프: `generic_ev_concept_96_v1`의 6x16 상하좌우 인접 관계
- 출력: 셀별 4단계 위험 확률 및 단계
- 운영 추론: NumPy (PyTorch를 재생 워커 메모리에 로드하지 않음)
- 최종 안전 판정: 변경하지 않음 (`final_risk_level`은 BMS Supervisor가 담당)
- 미래 온도 예측: 하지 않음

`manifest.json`의 검증 수치는 생성된 디지털 트윈 시나리오 내부 분할 결과입니다.
실차 안전 성능이나 외부 데이터 일반화 성능으로 해석하면 안 되며, 이 모델은 현재
프로토타입 3D 시각화 보강 용도입니다.

재학습:

```powershell
.\.venv\Scripts\python.exe scripts\train_cell_risk_gnn.py `
  --scenario-dir runtime\scenarios `
  --output-dir model_bundles\cell_risk_gnn_v1
```
