# -*- coding: utf-8 -*-
"""
멀티에이전트 파이프라인 — 게이트형 순차 구조

  [사용자 입력]
       │
       ▼
  Agent 1: Safety Guard  (fire_risk_model)
       │  화재/열폭주 위험 실시간 감지
       ├── 위험(Risk > Threshold) ──▶ 즉시 폐기 경보 (파이프라인 중단)
       ▼ 안전(Safe)
  Agent 2: Status Classifier  (reuse_model)
       │  SOH 등급(1/2/3등급) 판정
       ▼
  Agent 3: Value Assessor  (rul_model + valuation.py)
       │  잔여수명 예측 & 매각가 산정
       ▼
  Claude 종합 — 세 에이전트의 결과를 하나의 진단 리포트로 작성

⚠️ 판별(Agent1~3)은 전부 결정론적 코드다. LLM 호출은 마지막 종합 1회뿐이라
   같은 입력이면 등급·가격은 항상 같고, 리포트 문장만 매번 자연어로 생성된다.
"""
import os
import json

import numpy as np


def _fire_defaults_and_features(fire_bundle, fire_defaults):
    return fire_bundle["features"], fire_defaults


def run_agent1_safety_guard(fire_vals: dict, *, fire_model, fire_model_bin,
                            fire_features, fire_defaults,
                            threshold: float = 0.70) -> dict:
    """Agent 1: Safety Guard — fire_risk_model로 화재/열폭주 위험을 실시간 감지한다.

    보수적 임계값(기본 0.70)을 써서 오탐(거짓 경보)을 최소화한다. 이 임계값에서
    학습 데이터 기준 거짓 경보 0%, 실제 위험 검출 87% 수준이다(원본 rul/app.py
    FIRE_SENSITIVITY 참고). 위험으로 판정되면 뒤 단계(Agent2/3)를 실행하지 않는다.
    """
    v = dict(fire_defaults)
    v.update(fire_vals)
    X = np.array([[v[f] for f in fire_features]])

    stage = int(fire_model.predict(X)[0])
    fire_prob = float(fire_model_bin.predict_proba(X)[0, 1])  # stage>=4 위험 확률
    risky = fire_prob >= threshold

    return {
        "agent": "Agent 1: Safety Guard",
        "모델": "fire_risk_model",
        "예측_열폭주단계": stage,
        "화재위험확률_퍼센트": round(fire_prob * 100, 1),
        "임계값_퍼센트": round(threshold * 100, 1),
        "판정": "🚨 위험 — 파이프라인 중단" if risky else "🟢 안전 — 다음 단계 진행",
        "위험": risky,
    }


def run_agent2_status_classifier(feat: dict, *, reuse_model, reuse_features) -> dict:
    """Agent 2: Status Classifier — reuse_model로 SOH 등급을 판정한다."""
    X = np.array([[feat[f] for f in reuse_features]])
    grade = str(reuse_model.predict(X)[0])
    proba = reuse_model.predict_proba(X)[0]
    classes = list(reuse_model.classes_)
    conf = float(proba[classes.index(grade)])

    return {
        "agent": "Agent 2: Status Classifier",
        "모델": "reuse_model",
        "판정등급": grade,
        "신뢰도_퍼센트": round(conf * 100, 1),
        "등급별_확률": {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)},
    }


def run_agent3_value_assessor(feat: dict, grade: str, capacity_kwh: float,
                               health_pct: float, *, rul_model, rul_features,
                               valuation_mod, condition: float) -> dict:
    """Agent 3: Value Assessor — rul_model(잔여수명) + valuation.py(매각가)로
    이 배터리의 경제적 가치를 산정한다."""
    X = np.array([[feat[f] for f in rul_features]])
    rul = max(0.0, float(rul_model.predict(X)[0]))

    offers = valuation_mod.estimate_offers(grade, capacity_kwh, condition)
    best = offers[0] if offers else None

    return {
        "agent": "Agent 3: Value Assessor",
        "모델": "rul_model + valuation.py",
        "예측_잔여수명_사이클": round(rul, 1),
        "건강도_퍼센트": round(health_pct, 1),
        "최고제안처": best["매입처"] if best else None,
        "매입경로": best["단가대"] if best else None,
        "제안가": valuation_mod.won(best["제안가_원"]) if best else None,
        "매입처_수": len(offers),
    }


SYSTEM = """당신은 배터리 진단 리포트 작성자입니다.
아래는 3개의 에이전트가 순차로 실행되어 얻은 결과입니다(당신이 계산한 것이 아니라
이미 계산된 결과입니다). 이 결과만 근거로 삼아 한국어 마크다운 진단 리포트를 쓰십시오.

파이프라인 구조:
  Agent 1(Safety Guard, fire_risk_model) → 위험 시 즉시 중단
  Agent 2(Status Classifier, reuse_model) → SOH 등급 판정
  Agent 3(Value Assessor, rul_model+valuation) → 잔여수명·매각가 산정

리포트 형식:
## 종합 진단
파이프라인 3단계 결과를 한 문단으로 요약합니다.

## 단계별 결과
Agent 1/2/3 각각의 핵심 수치를 표나 목록으로 제시합니다.

## 권장 조치
우선순위대로 제시합니다.

## 유의사항
가격은 ML 학습 결과가 아니라 공개 벤치마크 기반 추정치임을 밝힙니다.

과장하지 말고, 주어진 수치 외의 것은 추측하지 마십시오."""


def run_pipeline(sensor_values: dict, fire_values: dict, capacity_kwh: float,
                 *, models: dict, question: str = "",
                 api_key: str | None = None) -> dict:
    """3-Agent 게이트 파이프라인 실행 + Claude 종합 리포트 생성.

    Args:
        sensor_values: RUL/SOH 판별용 7개 원본 센서값
        fire_values: Safety Guard용 8개 화재 센서값
        capacity_kwh: 배터리 팩 용량
        models: {"derive_features", "fire_model", "fire_model_bin", "fire_features",
                 "fire_defaults", "reuse_model", "reuse_features", "rul_model",
                 "rul_features", "valuation_mod", "compute_indicators"} 를 담은 dict
        question: 사용자 추가 질문(선택)
        api_key: Anthropic API 키. None이면 환경변수 사용.
    Returns:
        {"agent1", "agent2"(optional), "agent3"(optional), "report", "stopped_at"}
    """
    derive_features = models["derive_features"]
    feat = derive_features(sensor_values)

    result = {"stopped_at": None}

    agent1 = run_agent1_safety_guard(
        fire_values,
        fire_model=models["fire_model"], fire_model_bin=models["fire_model_bin"],
        fire_features=models["fire_features"], fire_defaults=models["fire_defaults"],
    )
    result["agent1"] = agent1

    if agent1["위험"]:
        result["stopped_at"] = "agent1"
        result["report"] = (
            f"## 🚨 즉시 폐기 및 경보\n\n"
            f"**Agent 1: Safety Guard**가 화재/열폭주 위험을 감지해 파이프라인을 "
            f"중단했습니다.\n\n"
            f"- 예측 열폭주 단계: **{agent1['예측_열폭주단계']}단계**\n"
            f"- 화재 위험 확률: **{agent1['화재위험확률_퍼센트']}%** "
            f"(경보 임계값 {agent1['임계값_퍼센트']}%)\n\n"
            f"⚠️ 이 배터리는 SOH 등급 판정·매각가 산정 대상이 아닙니다. "
            f"즉시 격리·폐기 절차를 따르십시오."
        )
        return result

    rul_pred = max(0.0, float(models["rul_model"].predict(
        np.array([[feat[f] for f in models["rul_features"]]]))[0]))
    from_full_life = models.get("full_life_cycles", 1134)
    health_pct = min(100.0, rul_pred / from_full_life * 100)

    agent2 = run_agent2_status_classifier(
        feat, reuse_model=models["reuse_model"], reuse_features=models["reuse_features"])
    result["agent2"] = agent2

    ind = models["compute_indicators"](feat, rul_pred)
    condition = sum(ind.values()) / len(ind)

    agent3 = run_agent3_value_assessor(
        feat, agent2["판정등급"], capacity_kwh, health_pct,
        rul_model=models["rul_model"], rul_features=models["rul_features"],
        valuation_mod=models["valuation_mod"], condition=condition,
    )
    result["agent3"] = agent3

    # ---- Claude 1회 호출 — 세 에이전트 결과를 종합해 리포트 작성 ----
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "Anthropic API 키가 없습니다. 환경변수 ANTHROPIC_API_KEY를 설정하거나 "
            "대시보드에서 키를 입력하세요."
        )
    import anthropic
    client = anthropic.Anthropic(api_key=key)

    payload = json.dumps({"agent1": agent1, "agent2": agent2, "agent3": agent3},
                         ensure_ascii=False, indent=2)
    user_msg = f"[파이프라인 실행 결과]\n{payload}"
    if question.strip():
        user_msg += f"\n\n[추가 질문]\n{question.strip()}"

    resp = client.beta.messages.create(
        model="claude-opus-4-8",
        max_tokens=8000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "\n".join(b.text for b in resp.content if b.type == "text")
    result["report"] = text.strip()
    return result
