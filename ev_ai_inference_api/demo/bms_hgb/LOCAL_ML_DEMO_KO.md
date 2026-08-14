# BMS Hybrid HGB 로컬 단독 시연

이 명령은 FastAPI 배포 번들에 포함된 것과 동일한 30초·120초 HGB 모델,
피처 계산 및 확률 보정을 사용한다. Backend, JWT, EKS, RDS, Redis와 보고서
생성은 사용하지 않는다.

따라서 이 시연의 목적은 `Hybrid HGB 자체가 현재 위험 상태 4단계를 어떻게
분류하는지`를 화면으로 확인하는 것이다. 플랫폼 전체 연동과 Safety Fusion은
기존 `replay` 명령으로 별도 시연한다.

## 실행 위치

PowerShell에서 다음 폴더로 이동한다.

```powershell
cd "C:\Users\User\Documents\Codex\2026-07-23\5-5-05-610-1-5\work\ev-fastapi-dev_ch\ev_ai_inference_api"
$tfPython = "C:\Users\User\anaconda3\envs\tf_env\python.exe"
```

## 발표용 실행

5배속이면 전체 250초 논리 시계열을 실제 약 50초 동안 보여준다.

```powershell
& $tfPython -m app.bms_demo local-ml `
  --dataset "demo\bms_hgb\aihub_holdout_20250912005_demo.jsonl" `
  --speed 5 `
  --print-every 10 `
  --result "runtime\bms_hgb_local_ml_result.jsonl"
```

예상 단계 전환은 다음과 같다.

```text
warming_up@0 -> normal@29 -> caution@122 -> warning@171 -> emergency@209
```

출력의 `N/C/W/E`는 각각 정상·주의·경고·긴급의 보정 확률이다.

```text
seq=029 ... HGB=normal    N=0.999 C=0.000 W=0.000 E=0.000
seq=122 ... HGB=caution   N=0.349 C=0.602 W=0.034 E=0.015
seq=171 ... HGB=warning   N=0.001 C=0.001 W=0.987 E=0.011
seq=209 ... HGB=emergency N=0.002 C=0.003 W=0.306 E=0.688
```

## 즉시 사전 점검

발표 전에 기다리지 않고 모델 파일과 네 단계 출력을 확인하려면 `--speed 0`을
사용한다.

```powershell
& $tfPython -m app.bms_demo local-ml `
  --dataset "demo\bms_hgb\aihub_holdout_20250912005_demo.jsonl" `
  --speed 0 `
  --print-every 30 `
  --result "runtime\bms_hgb_local_ml_precheck.jsonl"
```

마지막에 아래 문장이 나오면 네 단계가 모두 확인된 것이다.

```text
all four ML stages observed: True
```

## 플랫폼 통합 시연용 연속 데이터

기존 단계별 압축본 대신 validation 실험 `20250825004`의 연속 1 Hz 구간을
사용한다. 이 파일은 120초 워밍업부터 긴급 진입 직후까지 586행이며,
10배속 재생 시 약 59초가 걸린다.

```powershell
& $tfPython -m app.bms_demo replay `
  --dataset "demo\bms_hgb\aihub_validation_20250825004_continuous_demo.jsonl" `
  --backend-url $backend `
  --car-id $carId `
  --charging-session-id $chargingSessionId `
  --speed 10 `
  --result "runtime\bms_hgb_continuous_backend_result.jsonl"
```

이 명령은 기존 통합 시연과 동일하게 JWT·차량·실제 충전 세션이 필요하다.
실험 선택 근거와 한계는 `continuous_manifest.json`에 기록되어 있다.

## 해석 시 주의사항

- `local-ml`은 순수 Hybrid HGB 시연이다. 물리 규칙과 최종 Safety Fusion은
  결과 파일에 감사용으로만 기록하고 콘솔 전면에는 ML 결과를 표시한다.
- JSONL의 원본 단계는 모델 입력에 전달하지 않는다.
- 입력은 AI-Hub 통제 실험 신호를 96셀 BMS 계약으로 변환한 발표용 어댑터다.
  실제 차량 96셀 측정값 또는 새로운 성능 평가셋이 아니다.
- 이 결과는 기존 검증 성능을 대신하는 지표가 아니라 동작 시연이다.
