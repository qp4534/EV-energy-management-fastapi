# -*- coding: utf-8 -*-
"""
배터리 진단 멀티에이전트 파이프라인 — FastAPI 백엔드

⑥ 대시보드 탭(app.py)과 같은 파이프라인(pipeline_agent.py)을 그대로 재사용한다.
프론트(Streamlit이든, 팀의 자체 웹앱이든)가 이 서버를 호출해 모델 결과를 받는다.

실행:
    uvicorn fastapi_app:app --host 0.0.0.0 --port 8000

환경변수:
    DEEPSEEK_API_KEY_NH  — Agent1~3 종합 리포트(DeepSeek 1회 호출)에 사용.
                           서버 쪽에만 두고 클라이언트에는 절대 노출하지 않는다.

엔드포인트:
    GET  /health                 — 헬스체크 + 로드된 모델 정보
    POST /agents/safety-guard    — Agent 1만 단독 호출 (화재 위험)
    POST /agents/status-classifier — Agent 2만 단독 호출 (SOH 등급)
    POST /agents/value-assessor   — Agent 3만 단독 호출 (RUL + 매각가)
    POST /pipeline                — Agent 1→2→3 게이트 파이프라인 (+ 선택적 DeepSeek 종합)
    POST /report/pdf              — 매도 제안서 PDF 생성 (Agent 1→2→3 + 매입처 매칭 + PDF 렌더링)
    POST /report/pdf/full         — 이미 계산된 진단결과로 매입처 매칭+경제성까지 포함한 정식 PDF
    POST /report/pdf/from-view    — 화면에 뜬 값 그대로만 렌더링(매입처 매칭/경제성 없음, 단순 버전)
"""
import os
import io
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import pipeline_agent as PIPE
import valuation as VAL
import economics as ECO
import pdf_report as PDF
import buyer_lookup as BUYER

SENSOR_COLS = ["discharge_time_s", "decrement_36_34v_s", "max_volt_dischg_v",
               "min_volt_charg_v", "time_at_415v_s", "time_cc_s", "charging_time_s"]
FIRE_COLS = ["surface_temp_c", "avg_battery_temp_c", "voltage_v", "pressure_kpa",
             "temp_rise_c_per_sec", "total_gas_ppm", "co_ppm", "soc_pct"]

HERE = os.path.dirname(os.path.abspath(__file__))

_MODELS: dict = {}


def _derive_features(v: dict) -> dict:
    d = dict(v)
    d["volt_span_v"] = d["max_volt_dischg_v"] - d["min_volt_charg_v"]
    d["cc_ratio"] = d["time_cc_s"] / (d["charging_time_s"] + 1e-9)
    d["dischg_per_charge"] = d["discharge_time_s"] / (d["charging_time_s"] + 1e-9)
    return d


def _nz(v, lo, hi):
    return float(np.clip((v - lo) / (hi - lo + 1e-9), 0.0, 1.0))


def _require_model_file(filename: str) -> str:
    """rul_model_B_no_cycle.joblib(304MB)·fire_risk_model.joblib(125MB)는
    GitHub 100MB 제한 때문에 이 저장소에 커밋되어 있지 않다(.gitignore 처리).
    README.md 안내대로 이 디렉터리에 파일을 직접 받아 둔 뒤에 서버를 띄워야 한다."""
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        raise RuntimeError(
            f"모델 파일이 없습니다: {filename}\n"
            f"ai-services/rul-diagnosis/README.md의 '모델 파일 준비' 절차를 따라 "
            f"이 디렉터리({HERE})에 {filename}을 받아 둔 뒤 다시 실행하세요."
        )
    return path


def _load_all_models() -> dict:
    """서버 기동 시 1회만 로드. joblib 모델은 수백MB라 매 요청마다 로드하면 안 된다."""
    rul_b = joblib.load(_require_model_file("rul_model_B_no_cycle.joblib"))
    reuse_b = joblib.load(os.path.join(HERE, "reuse_model.joblib"))
    fire_b = joblib.load(_require_model_file("fire_risk_model.joblib"))
    with open(os.path.join(HERE, "fire_risk_defaults.json"), encoding="utf-8") as f:
        fire_defaults = json.load(f)
    with open(os.path.join(HERE, "reuse_match_config.json"), encoding="utf-8") as f:
        match = json.load(f)
    norm, full_life = match["NORM"], match["FULL_LIFE"]

    def compute_indicators(feat: dict, rul_pred: float) -> dict:
        return {
            "life": float(np.clip(rul_pred / full_life, 0.0, 1.0)),
            "capacity": _nz(feat["discharge_time_s"],
                            norm["discharge_time_s"]["p5"], norm["discharge_time_s"]["p95"]),
            "charge": _nz(feat["cc_ratio"], norm["cc_ratio"]["p5"], norm["cc_ratio"]["p95"]),
            "stability": _nz(feat["decrement_36_34v_s"],
                             norm["decrement_36_34v_s"]["p5"], norm["decrement_36_34v_s"]["p95"]),
        }

    return {
        "derive_features": _derive_features,
        "fire_model": fire_b["model"], "fire_model_bin": fire_b["model_binary"],
        "fire_features": fire_b["features"], "fire_defaults": fire_defaults,
        "reuse_model": reuse_b["model"], "reuse_features": reuse_b["features"],
        "rul_model": rul_b["model"], "rul_features": rul_b["features"],
        "valuation_mod": VAL, "compute_indicators": compute_indicators,
        "full_life_cycles": 1134,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _MODELS.update(_load_all_models())
    yield
    _MODELS.clear()


app = FastAPI(
    title="배터리 진단 멀티에이전트 파이프라인 API",
    description="Agent1(Safety Guard) → Agent2(Status Classifier) → Agent3(Value Assessor)",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# 요청/응답 스키마
# ------------------------------------------------------------------
class SensorValues(BaseModel):
    """RUL/SOH 판별용 원본 7개 센서값 — RAW_INPUTS와 동일"""
    discharge_time_s: float = Field(..., examples=[1156.62])
    decrement_36_34v_s: float = Field(..., examples=[318.86])
    max_volt_dischg_v: float = Field(..., examples=[3.86])
    min_volt_charg_v: float = Field(..., examples=[3.66])
    time_at_415v_s: float = Field(..., examples=[1798.09])
    time_cc_s: float = Field(..., examples=[2528.38])
    charging_time_s: float = Field(..., examples=[7358.62])


class FireValues(BaseModel):
    """Safety Guard용 8개 화재 센서값 — FIRE_INPUTS와 동일"""
    surface_temp_c: float = Field(..., examples=[31.4])
    avg_battery_temp_c: float = Field(..., examples=[39.9])
    voltage_v: float = Field(..., examples=[4.19])
    pressure_kpa: float = Field(..., examples=[12.57])
    temp_rise_c_per_sec: float = Field(..., examples=[0.0])
    total_gas_ppm: float = Field(..., examples=[18.8])
    co_ppm: float = Field(..., examples=[0.0])
    soc_pct: float = Field(..., examples=[50.0])


class PipelineRequest(BaseModel):
    sensor_values: SensorValues
    fire_values: FireValues
    capacity_kwh: float = Field(64.0, gt=0)
    question: str = ""
    include_report: bool = Field(
        True, description="False로 주면 DeepSeek 종합 호출을 건너뛰고 Agent1~3 "
                          "수치만 즉시 반환합니다(지연시간 없음, API 키 불필요).")


# ------------------------------------------------------------------
# 엔드포인트
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_models": ["rul_model", "reuse_model", "fire_risk_model"],
        "deepseek_key_configured": bool(os.environ.get("DEEPSEEK_API_KEY_NH")),
    }


@app.post("/agents/safety-guard")
def safety_guard(fire_values: FireValues):
    """Agent 1 단독 호출 — 화재/열폭주 위험만 판정한다."""
    return PIPE.run_agent1_safety_guard(
        fire_values.model_dump(),
        fire_model=_MODELS["fire_model"], fire_model_bin=_MODELS["fire_model_bin"],
        fire_features=_MODELS["fire_features"], fire_defaults=_MODELS["fire_defaults"],
    )


@app.post("/agents/status-classifier")
def status_classifier(sensor_values: SensorValues):
    """Agent 2 단독 호출 — SOH 등급만 판정한다."""
    feat = _derive_features(sensor_values.model_dump())
    return PIPE.run_agent2_status_classifier(
        feat, reuse_model=_MODELS["reuse_model"], reuse_features=_MODELS["reuse_features"])


@app.post("/agents/value-assessor")
def value_assessor(sensor_values: SensorValues, grade: str, capacity_kwh: float = 64.0):
    """Agent 3 단독 호출 — grade(예: '1등급')를 이미 알고 있을 때 가치만 평가한다."""
    if grade not in ("1등급", "2등급", "3등급"):
        raise HTTPException(400, "grade는 '1등급'|'2등급'|'3등급' 중 하나여야 합니다.")
    feat = _derive_features(sensor_values.model_dump())
    rul = max(0.0, float(_MODELS["rul_model"].predict(
        np.array([[feat[f] for f in _MODELS["rul_features"]]]))[0]))
    health_pct = min(100.0, rul / _MODELS["full_life_cycles"] * 100)
    ind = _MODELS["compute_indicators"](feat, rul)
    condition = sum(ind.values()) / len(ind)
    return PIPE.run_agent3_value_assessor(
        feat, grade, capacity_kwh, health_pct,
        rul_model=_MODELS["rul_model"], rul_features=_MODELS["rul_features"],
        valuation_mod=VAL, condition=condition,
    )


@app.post("/pipeline")
def pipeline(req: PipelineRequest):
    """Agent1→2→3 게이트 파이프라인 전체 실행.

    include_report=False면 DeepSeek를 호출하지 않고 세 에이전트의 수치만 반환한다
    (백엔드에서 EV 재제조/ESS 재사용 가능 여부를 즉시 판단하는 용도로 적합).
    include_report=True면 마지막에 DeepSeek이 자연어 리포트까지 작성한다
    (서버 환경변수 DEEPSEEK_API_KEY_NH 필요).
    """
    if req.include_report and not os.environ.get("DEEPSEEK_API_KEY_NH"):
        raise HTTPException(
            500, "DEEPSEEK_API_KEY_NH가 서버에 설정되지 않았습니다. "
                "include_report=false로 요청하거나 서버 환경변수를 설정하세요.")

    if not req.include_report:
        # DeepSeek 호출 없이 결정론적 3단계만 실행 — 빠르고 API 키도 불필요
        feat = _derive_features(req.sensor_values.model_dump())
        agent1 = PIPE.run_agent1_safety_guard(
            req.fire_values.model_dump(),
            fire_model=_MODELS["fire_model"], fire_model_bin=_MODELS["fire_model_bin"],
            fire_features=_MODELS["fire_features"], fire_defaults=_MODELS["fire_defaults"],
        )
        if agent1["위험"]:
            return {"stopped_at": "agent1", "agent1": agent1}

        rul = max(0.0, float(_MODELS["rul_model"].predict(
            np.array([[feat[f] for f in _MODELS["rul_features"]]]))[0]))
        health_pct = min(100.0, rul / _MODELS["full_life_cycles"] * 100)
        agent2 = PIPE.run_agent2_status_classifier(
            feat, reuse_model=_MODELS["reuse_model"], reuse_features=_MODELS["reuse_features"])
        ind = _MODELS["compute_indicators"](feat, rul)
        condition = sum(ind.values()) / len(ind)
        agent3 = PIPE.run_agent3_value_assessor(
            feat, agent2["판정등급"], req.capacity_kwh, health_pct,
            rul_model=_MODELS["rul_model"], rul_features=_MODELS["rul_features"],
            valuation_mod=VAL, condition=condition,
        )
        return {"stopped_at": None, "agent1": agent1, "agent2": agent2, "agent3": agent3}

    try:
        return PIPE.run_pipeline(
            sensor_values=req.sensor_values.model_dump(),
            fire_values=req.fire_values.model_dump(),
            capacity_kwh=req.capacity_kwh,
            models=_MODELS,
            question=req.question,
            api_key=None,  # 서버 환경변수 DEEPSEEK_API_KEY_NH 사용
        )
    except RuntimeError as e:
        raise HTTPException(500, str(e))


class PdfReportRequest(BaseModel):
    sensor_values: SensorValues
    fire_values: FireValues
    capacity_kwh: float = Field(64.0, gt=0)
    buyer_index: int = Field(
        0, ge=0, description="estimate_offers() 결과 중 몇 번째 매입처로 제안서를 "
                             "만들지 (0 = 최고 제안가). 관리자 웹에서 매입처를 고르게 "
                             "하려면 먼저 /pipeline으로 목록을 받아 인덱스를 골라야 함.")
    chemistry: str = Field("NMC811", description="배터리 화학조성 — 탄소 절감량 계산에 사용")
    new_price_krw: Optional[float] = Field(
        None, description="신품 팩 단가(원/kWh). 생략하면 economics.py 기본값(BNEF 국제시세) 사용.")


@app.post("/report/pdf")
def report_pdf(req: PdfReportRequest):
    """매도 제안서 PDF 생성.

    /pipeline과 같은 Agent1→2→3 계산을 거친 뒤, 매입처를 하나 골라
    pdf_report.build_pdf()로 렌더링해 PDF 바이트를 그대로 스트리밍한다.
    화재 위험 게이트(Agent1)에 걸리면 제안서를 만들지 않고 409를 반환한다.
    """
    feat = _derive_features(req.sensor_values.model_dump())

    agent1 = PIPE.run_agent1_safety_guard(
        req.fire_values.model_dump(),
        fire_model=_MODELS["fire_model"], fire_model_bin=_MODELS["fire_model_bin"],
        fire_features=_MODELS["fire_features"], fire_defaults=_MODELS["fire_defaults"],
    )
    if agent1["위험"]:
        raise HTTPException(409, {
            "message": "화재 위험 게이트에서 중단 — 즉시 폐기 대상이라 제안서를 만들지 않습니다.",
            "agent1": agent1,
        })

    rul = max(0.0, float(_MODELS["rul_model"].predict(
        np.array([[feat[f] for f in _MODELS["rul_features"]]]))[0]))
    health_pct = min(100.0, rul / _MODELS["full_life_cycles"] * 100)
    agent2 = PIPE.run_agent2_status_classifier(
        feat, reuse_model=_MODELS["reuse_model"], reuse_features=_MODELS["reuse_features"])
    grade = agent2["판정등급"]
    ind = _MODELS["compute_indicators"](feat, rul)
    condition = sum(ind.values()) / len(ind)

    offers = VAL.estimate_offers(grade, req.capacity_kwh, condition)
    if not offers:
        raise HTTPException(422, f"'{grade}' 등급을 매입해줄 매입처가 없습니다.")
    if req.buyer_index >= len(offers):
        raise HTTPException(400, f"buyer_index가 범위를 벗어났습니다. 매입처는 {len(offers)}곳입니다.")
    chosen = offers[req.buyer_index]
    # 실시간 검색으로 매입처 최신 정보를 찾을 수 있으면 정적 문구를 덮어쓴다.
    # 검색 실패/키 없음이면 valuation.py의 기존 정적 "확인된 사실" 문구를 그대로 씀.
    live_fact = BUYER.fetch_buyer_disclosure(chosen["매입처"])
    if live_fact:
        chosen = {**chosen, "확인된_사실": live_fact}

    eco_kwargs = {"chemistry": req.chemistry, "path": chosen["단가대"]}
    if req.new_price_krw is not None:
        eco_kwargs["new_price_krw"] = req.new_price_krw
    eco = ECO.compute(req.capacity_kwh, chosen["제안가_원"], grade, health_pct, **eco_kwargs)

    pdf_bytes = PDF.build_pdf(
        buyer=chosen, capacity_kwh=req.capacity_kwh, grade=grade,
        rul_cycles=rul, health_pct=health_pct, indicators=ind,
        full_life=_MODELS["full_life_cycles"], won=VAL.won, eco=eco,
        fire_note=agent1.get("판정", ""),
    )

    # Content-Disposition 헤더는 latin-1만 허용한다 - 한글 파일명은 RFC 5987
    # filename*=UTF-8''... 형식으로 별도 인코딩해야 한다.
    from urllib.parse import quote
    fname = quote(f"매도제안서_{chosen['매입처'].split()[0]}_{req.capacity_kwh:g}kWh.pdf")
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=battery_proposal.pdf; "
                                        f"filename*=UTF-8''{fname}"},
    )


# ------------------------------------------------------------------
# ERD 연동용 어댑터 — BATTERY_PASSPORT / BATTERY_DIAGNOSIS_METRICS 컬럼명에 맞춘 응답.
# 백엔드가 이 엔드포인트 결과를 그대로 UPDATE 문에 매핑해서 쓸 수 있게 하기 위한 것.
# Agent1~3 자체(위 /pipeline)와 판정 로직은 동일하고, 출력 키만 바꾼다.
# ------------------------------------------------------------------
_GRADE_TO_LEVEL = {"1등급": "1", "2등급": "2", "3등급": "3"}
_GRADE_TO_REUSE_STATUS = {"1등급": "양호", "2등급": "노후", "3등급": "수명말기"}


class ErdDiagnoseRequest(BaseModel):
    sensor_values: SensorValues
    fire_values: FireValues
    capacity_kwh: float = Field(64.0, gt=0)


@app.post("/diagnose/erd")
def diagnose_erd(req: ErdDiagnoseRequest):
    """Agent1→2→3을 실행하고(DeepSeek 종합 없이) ERD 컬럼명에 맞춰 반환한다.

    반환 필드는 BATTERY_PASSPORT의 battery_level/reuse_status/grade_detail/
    reliability_score/rul, BATTERY_DIAGNOSIS_METRICS의 4개 점수와 대응된다.

    ⚠️ 주의할 점 두 가지(백엔드팀과 확정 필요):
    1. `rul_raw_cycles`는 모델이 예측한 원본 잔여 사이클 수(보통 수백~수천)다.
       ERD의 BATTERY_PASSPORT.rul은 NUMERIC(4,1)(최대 999.9)이라 그대로 넣으면
       자릿수가 넘칠 수 있다. 스케일링 방식(예: 연 단위 환산, 정규화 0~10)을
       정해서 `rul`(변환값)을 쓰거나 컬럼 타입 조정을 논의해야 한다.
    2. `reuse_status`는 등급→상태 3단계 매핑(1등급=양호/2등급=노후/3등급=수명말기)의
       임시 규칙이다. 실제 운영 기준이 확정되면 이 매핑만 바꾸면 된다.
    """
    feat = _derive_features(req.sensor_values.model_dump())
    agent1 = PIPE.run_agent1_safety_guard(
        req.fire_values.model_dump(),
        fire_model=_MODELS["fire_model"], fire_model_bin=_MODELS["fire_model_bin"],
        fire_features=_MODELS["fire_features"], fire_defaults=_MODELS["fire_defaults"],
    )
    if agent1["위험"]:
        return {
            "battery_level": "3",
            "reuse_status": "수명말기",
            "grade_detail": None,
            "reliability_score": None,
            "rul": 0.0,
            "rul_raw_cycles": 0.0,
            "remaining_life_score": 0,
            "discharge_power_score": None,
            "charge_health_score": None,
            "voltage_stability_score": None,
            "위험_안내": "Agent1 화재/열폭주 위험 감지 — 즉시 격리·폐기 대상, 등급 판정 생략",
        }

    rul_raw = max(0.0, float(_MODELS["rul_model"].predict(
        np.array([[feat[f] for f in _MODELS["rul_features"]]]))[0]))
    health_pct = min(100.0, rul_raw / _MODELS["full_life_cycles"] * 100)
    agent2 = PIPE.run_agent2_status_classifier(
        feat, reuse_model=_MODELS["reuse_model"], reuse_features=_MODELS["reuse_features"])
    ind = _MODELS["compute_indicators"](feat, rul_raw)
    condition = sum(ind.values()) / len(ind)
    agent3 = PIPE.run_agent3_value_assessor(
        feat, agent2["판정등급"], req.capacity_kwh, health_pct,
        rul_model=_MODELS["rul_model"], rul_features=_MODELS["rul_features"],
        valuation_mod=VAL, condition=condition,
    )

    grade = agent2["판정등급"]
    return {
        "battery_level": _GRADE_TO_LEVEL.get(grade, grade),
        "reuse_status": _GRADE_TO_REUSE_STATUS.get(grade, grade),
        "grade_detail": agent3["매입경로"],
        "reliability_score": agent2["신뢰도_퍼센트"],
        "rul": round(min(rul_raw, 999.9), 1),  # 자릿수 초과 임시 clip — 위 주의사항 1 참고
        "rul_raw_cycles": round(rul_raw, 1),
        "remaining_life_score": int(round(health_pct)),
        "discharge_power_score": int(round(ind["capacity"] * 100)),
        "charge_health_score": int(round(ind["charge"] * 100)),
        "voltage_stability_score": int(round(ind["stability"] * 100)),
    }


class IndicatorsView(BaseModel):
    life: float
    capacity: float
    charge: float
    stability: float


class PdfFullRequest(BaseModel):
    """이미 계산되어 저장된 진단 결과(등급/RUL/건강도/세부지표)를 받아, 원본 센서값 없이도
    /report/pdf와 똑같이 매입처 매칭(estimate_offers)·경제성(economics.compute)까지 포함한
    정식 문서를 만든다. Agent1~3을 다시 돌리지 않는다 - 화면에 이미 뜬 숫자를 근거로만 쓴다."""
    capacity_kwh: float = Field(..., gt=0)
    grade: str
    rul_cycles: float
    full_life: float
    health_pct: float
    indicators: IndicatorsView
    buyer_index: int = Field(0, ge=0)
    chemistry: str = "NMC811"
    new_price_krw: float | None = None
    chosen_buyer: dict | None = Field(
        None, description="화면(\"잔존가치/판매처\" 탭)에서 이미 선택된 매입처를 그대로 "
                          "쓰고 싶을 때 넘긴다 - buyer_index로 static BUYERS를 다시 찾는 "
                          "대신, estimate_offers()/discover_buyers()가 반환한 offer 객체를 "
                          "그대로 전달하면 된다(화면과 PDF의 매입처가 어긋나지 않게 하기 위함, "
                          "특히 실시간 검색으로 찾은 매입처를 골랐을 때 필요).")
    reasons: list[str] = Field(
        default_factory=list, description="화면에 이미 표시된 \"귀사에 적합한 이유\" "
                                          "추가 문구 - PDF 3번 섹션에 매입처 왜(why) 아래로 "
                                          "그대로 덧붙여, 화면과 PDF 내용이 같게 한다.")


@app.post("/report/pdf/full")
def report_pdf_full(req: PdfFullRequest):
    """매입처 매칭 + 경제성 계산까지 포함한 정식 매도 제안서 PDF (실제 회사 제출용 수준)."""
    if req.grade not in ("1등급", "2등급", "3등급"):
        raise HTTPException(400, "grade는 '1등급'|'2등급'|'3등급' 중 하나여야 합니다.")

    indicators = req.indicators.model_dump()
    condition = sum(indicators.values()) / len(indicators)

    if req.chosen_buyer:
        # 화면에서 이미 고른 매입처(실시간 검색 결과 포함)를 그대로 쓴다 - buyer_index로
        # static BUYERS를 다시 찾지 않는다(검색 매입처는애초에 static 목록에 없음).
        chosen = req.chosen_buyer
    else:
        offers = VAL.estimate_offers(req.grade, req.capacity_kwh, condition)
        if not offers:
            raise HTTPException(422, f"'{req.grade}' 등급을 매입해줄 매입처가 없습니다.")
        if req.buyer_index >= len(offers):
            raise HTTPException(400, f"buyer_index가 범위를 벗어났습니다. 매입처는 {len(offers)}곳입니다.")
        chosen = offers[req.buyer_index]

    # 실시간 검색(Serper)+요약(DeepSeek)으로 매입처 최신 정보를 찾을 수 있으면 "확인된 사업
    # 영역"뿐 아니라 "귀사 적합성"(section 4, buyer["왜"])도 이 실제 근거로 덮어쓴다 - 정적
    # 문구/검색 요약 role 한 줄보다 신뢰도가 높다. 같은 호출 결과를 두 필드에 재사용해서
    # DeepSeek을 두 번 부르지 않는다(비용 절감). chosen_buyer 경로(화면에서 이미 실시간
    # 검색으로 고른 매입처)에도 똑같이 적용 - discover_buyers()의 "왜"는 role 한 줄짜리라
    # 이걸로 더 근거 있는 문장으로 보강한다. 키가 없거나 검색/요약이 실패하면 None ->
    # 기존 문구 그대로 폴백.
    live_fact = BUYER.fetch_buyer_disclosure(chosen["매입처"])
    if live_fact:
        chosen = {**chosen, "확인된_사실": live_fact, "왜": live_fact}

    eco_kwargs = {"chemistry": req.chemistry, "path": chosen["단가대"]}
    if req.new_price_krw is not None:
        eco_kwargs["new_price_krw"] = req.new_price_krw
    eco = ECO.compute(req.capacity_kwh, chosen["제안가_원"], req.grade, req.health_pct, **eco_kwargs)

    pdf_bytes = PDF.build_pdf(
        buyer=chosen, capacity_kwh=req.capacity_kwh, grade=req.grade,
        rul_cycles=req.rul_cycles, health_pct=req.health_pct, indicators=indicators,
        full_life=req.full_life, won=VAL.won, eco=eco, extra_reasons=req.reasons,
    )

    from urllib.parse import quote
    fname = quote(f"매도제안서_{chosen['매입처'].split()[0]}_{req.capacity_kwh:g}kWh.pdf")
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=battery_proposal.pdf; "
                                        f"filename*=UTF-8''{fname}"},
    )


class HealthMetricView(BaseModel):
    label: str
    score: str


class PdfFromViewRequest(BaseModel):
    """관리자 웹(BatteryDiagnosis.jsx)이 화면에 이미 표시한 값 그대로. Agent1~3을
    다시 돌리지 않으므로 화면 숫자와 PDF 숫자가 어긋날 일이 없다."""
    buyer_name: str = "매입 희망 기업"
    buyer_role: str = ""
    buyer_location: str = ""
    price_total_manwon: float
    unit_price_won: float
    negotiation_range: str
    price_grade_label: str
    price_note: str = ""
    grade: str
    remaining_cycle: float
    new_cycle: float
    health_score_pct: float
    health_metrics: list[HealthMetricView] = []
    diagnosis_note: str = ""
    reasons: list[str] = []
    cautions: list[str] = []


@app.post("/report/pdf/from-view")
def report_pdf_from_view(req: PdfFromViewRequest):
    """이미 계산되어 화면에 뜬 진단/제안 값을 그대로 PDF로 렌더링만 한다."""
    pdf_bytes = PDF.build_pdf_from_view(
        buyer_name=req.buyer_name, buyer_role=req.buyer_role, buyer_location=req.buyer_location,
        price_total_manwon=req.price_total_manwon, unit_price_won=req.unit_price_won,
        negotiation_range=req.negotiation_range, price_grade_label=req.price_grade_label,
        price_note=req.price_note, grade=req.grade, remaining_cycle=req.remaining_cycle,
        new_cycle=req.new_cycle, health_score_pct=req.health_score_pct,
        health_metrics=[m.model_dump() for m in req.health_metrics],
        diagnosis_note=req.diagnosis_note, reasons=req.reasons, cautions=req.cautions,
    )
    from urllib.parse import quote
    fname = quote(f"매도제안서_{req.buyer_name.split()[0]}.pdf")
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=battery_proposal.pdf; "
                                        f"filename*=UTF-8''{fname}"},
    )


class LiveOffersRequest(BaseModel):
    """"잔존가치/판매처" 탭 로드 시 매입처 목록을 가져오는 요청. 회사 자체를 검색으로
    찾아서 매입처 목록을 구성하되, 가격은 그대로 기존 PRICE_BANDS(BNEF/국내 낙찰가 등
    출처가 있는 벤치마크) 계산식으로 산정한다. Serper/DeepSeek 키는 서버 환경변수
    (SERPER_API_KEY_NH/DEEPSEEK_API_KEY_NH)로 자동 처리된다."""
    grade: str
    capacity_kwh: float = Field(..., gt=0)
    condition: float = Field(..., ge=0, le=1)


@app.post("/buyers/live-offers")
def buyers_live_offers(req: LiveOffersRequest):
    """매입처를 실시간 검색으로 찾아서(discover_buyers) 매칭·가격 계산까지 한 번에 돌려준다.
    검색이 실패하거나 키가 없으면 기존 고정 매입처 목록(valuation.BUYERS)으로 자동
    폴백한다 - 화면이 절대 빈 목록이 되지 않는다."""
    if req.grade not in ("1등급", "2등급", "3등급"):
        raise HTTPException(400, "grade는 '1등급'|'2등급'|'3등급' 중 하나여야 합니다.")

    discovered = BUYER.discover_buyers()
    offers = VAL.estimate_offers(
        req.grade, req.capacity_kwh, req.condition, buyers=discovered,
    )
    return {"live": discovered is not None, "offers": offers}


# ------------------------------------------------------------------
# 엑셀 일괄 업로드 — 관리자가 여러 배터리 행을 한 번에 올려 파이프라인을 돌린다.
#   DeepSeek 호출(리포트 작성)은 하지 않는다 — 수백 행을 한 번에 처리할 수 있으므로
#   행마다 LLM을 부르면 느리고 비용도 크다. Agent1~3 수치만 원본 표에 붙여 반환한다.
#   개별 행의 자연어 리포트가 필요하면 그 행만 /pipeline 으로 다시 호출하면 된다.
# ------------------------------------------------------------------
@app.post("/pipeline/batch-excel")
async def pipeline_batch_excel(
    file: UploadFile = File(..., description=(
        f"xlsx 파일. 필수 열: {', '.join(SENSOR_COLS)}. "
        f"선택 열(있으면 Agent1 화재 게이트도 실행): {', '.join(FIRE_COLS)}. "
        "capacity_kwh 열이 없으면 default_capacity_kwh 값을 모든 행에 적용한다.")),
    default_capacity_kwh: float = Form(64.0, gt=0),
    return_format: str = Form("json", description="'json' 또는 'xlsx'"),
):
    """엑셀 한 장 -> 행마다 Agent1(선택)→Agent2→Agent3 실행 -> 결과 열을 붙여 반환.

    - 필수 열이 하나라도 없으면 400과 함께 어떤 열이 빠졌는지 알려준다.
    - 화재 8개 열이 전부 있으면 Agent1을 실행해 위험 행은 Agent2/3를 건너뛴다.
      화재 열이 없으면 Agent1은 건너뛰고 '데이터없음'으로 표시한다(안전 가정 아님 —
      화재 여부를 아예 판단하지 않았다는 뜻이므로 별도 화재 점검이 필요하다).
    """
    if return_format not in ("json", "xlsx"):
        raise HTTPException(400, "return_format은 'json' 또는 'xlsx' 여야 합니다.")

    raw = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"엑셀 파일을 읽지 못했습니다: {e}")

    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            400, f"필수 열이 없습니다: {missing}. 필요한 열: {SENSOR_COLS}")

    has_fire = all(c in df.columns for c in FIRE_COLS)
    has_capacity_col = "capacity_kwh" in df.columns

    rows_out = []
    for i, row in df.iterrows():
        sensor_vals = {c: float(row[c]) for c in SENSOR_COLS}
        capacity = float(row["capacity_kwh"]) if has_capacity_col else default_capacity_kwh
        feat = _derive_features(sensor_vals)

        out = {"행번호": int(i) + 1}

        if has_fire:
            fire_vals = {c: float(row[c]) for c in FIRE_COLS}
            a1 = PIPE.run_agent1_safety_guard(
                fire_vals, fire_model=_MODELS["fire_model"],
                fire_model_bin=_MODELS["fire_model_bin"],
                fire_features=_MODELS["fire_features"],
                fire_defaults=_MODELS["fire_defaults"])
            out["Agent1_판정"] = a1["판정"]
            out["Agent1_화재위험확률(%)"] = a1["화재위험확률_퍼센트"]
            if a1["위험"]:
                out["Agent2_판정등급"] = None
                out["Agent3_제안가"] = None
                out["비고"] = "Agent1 게이트에서 중단 — 즉시 폐기 대상"
                rows_out.append(out)
                continue
        else:
            out["Agent1_판정"] = "화재 데이터 없음(미실행)"
            out["Agent1_화재위험확률(%)"] = None

        rul = max(0.0, float(_MODELS["rul_model"].predict(
            np.array([[feat[f] for f in _MODELS["rul_features"]]]))[0]))
        health_pct = min(100.0, rul / _MODELS["full_life_cycles"] * 100)
        a2 = PIPE.run_agent2_status_classifier(
            feat, reuse_model=_MODELS["reuse_model"],
            reuse_features=_MODELS["reuse_features"])
        ind = _MODELS["compute_indicators"](feat, rul)
        condition = sum(ind.values()) / len(ind)
        a3 = PIPE.run_agent3_value_assessor(
            feat, a2["판정등급"], capacity, health_pct,
            rul_model=_MODELS["rul_model"], rul_features=_MODELS["rul_features"],
            valuation_mod=VAL, condition=condition,
        )

        out["예측_잔여수명_사이클"] = a3["예측_잔여수명_사이클"]
        out["건강도(%)"] = a3["건강도_퍼센트"]
        out["Agent2_판정등급"] = a2["판정등급"]
        out["Agent2_신뢰도(%)"] = a2["신뢰도_퍼센트"]
        out["Agent3_최고제안처"] = a3["최고제안처"]
        out["Agent3_매입경로"] = a3["매입경로"]
        out["Agent3_제안가"] = a3["제안가"]
        out["비고"] = ""
        rows_out.append(out)

    result_df = pd.DataFrame(rows_out)

    if return_format == "json":
        return {
            "행수": len(result_df),
            "화재열_존재": has_fire,
            "결과": json.loads(result_df.to_json(orient="records", force_ascii=False)),
        }

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.concat([df.reset_index(drop=True), result_df.drop(columns=["행번호"])],
                 axis=1).to_excel(writer, index=False, sheet_name="진단결과")
    buf.seek(0)
    # Content-Disposition 헤더는 latin-1만 허용한다 — 한글 파일명은 RFC 5987
    # filename*=UTF-8''... 형식으로 별도 인코딩해야 한다(그냥 붙이면 500 에러).
    from urllib.parse import quote
    fname = quote("배터리진단결과.xlsx")
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=battery_diagnosis_result.xlsx; "
                                        f"filename*=UTF-8''{fname}"},
    )
