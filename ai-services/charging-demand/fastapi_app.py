# -*- coding: utf-8 -*-
"""
충전 수요 예측 — FastAPI 백엔드

app.py(Streamlit)와 같은 모델(demand_ca_model.joblib)을 그대로 재사용한다.
실행:
    uvicorn fastapi_app:app --host 127.0.0.1 --port 8001

엔드포인트:
    GET  /health           — 헬스체크
    POST /predict           — 특정 시각·요일·월·연도의 예상 충전 세션 수
    POST /predict/curve     — 요일·월·연도 고정, 0~23시 전체 곡선
"""
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "demand_ca_model.joblib")

_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(MODEL_PATH)
    _STATE["model"] = bundle["model"]
    _STATE["features"] = bundle["features"]
    _STATE["year_range"] = bundle["year_range"]
    _STATE["max_sessions"] = bundle["max_sessions"]
    yield
    _STATE.clear()


app = FastAPI(
    title="충전 수요 예측 API",
    description="팔로알토 실측 충전 데이터로 학습한 시간당 수요 예측 모델",
    version="1.0.0",
    lifespan=lifespan,
)


def _build_vector(hour: int, dow: int, month: int, year: int) -> np.ndarray:
    is_weekend = 1 if dow >= 5 else 0
    row = {"hour": hour, "dow": dow, "is_weekend": is_weekend,
           "month": month, "year": year}
    return np.array([[row[f] for f in _STATE["features"]]])


class DemandRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23, examples=[12])
    dow: int = Field(..., ge=0, le=6, description="0=월 ~ 6=일", examples=[2])
    month: int = Field(..., ge=1, le=12, examples=[6])
    year: int = Field(..., examples=[2019])


class CurveRequest(BaseModel):
    dow: int = Field(..., ge=0, le=6, description="0=월 ~ 6=일", examples=[2])
    month: int = Field(..., ge=1, le=12, examples=[6])
    year: int = Field(..., examples=[2019])


@app.get("/health")
def health():
    lo, hi = _STATE.get("year_range", (None, None))
    return {"status": "ok", "model": "demand_ca_model", "학습_연도범위": [lo, hi]}


@app.post("/predict")
def predict(req: DemandRequest):
    lo, hi = _STATE["year_range"]
    if not (lo <= req.year <= hi):
        raise HTTPException(
            400, f"year는 학습 범위({lo}~{hi}) 안이어야 합니다. 범위 밖 값은 신뢰도가 떨어집니다.")

    X = _build_vector(req.hour, req.dow, req.month, req.year)
    demand = max(0.0, float(_STATE["model"].predict(X)[0]))
    level_pct = min(1.0, demand / max(1, _STATE["max_sessions"]))

    if level_pct >= 0.6:
        level, note = "높음", "수요 집중 시간대. 충전기 증설·분산 유도가 필요합니다."
    elif level_pct >= 0.3:
        level, note = "보통", "평균 수준의 수요입니다."
    else:
        level, note = "낮음", "여유 있는 시간대입니다."

    return {
        "예상_충전세션수_시간당": round(demand, 1),
        "수요수준": level,
        "수요수준_비율": round(level_pct, 3),
        "설명": note,
    }


@app.post("/predict/curve")
def predict_curve(req: CurveRequest):
    lo, hi = _STATE["year_range"]
    if not (lo <= req.year <= hi):
        raise HTTPException(
            400, f"year는 학습 범위({lo}~{hi}) 안이어야 합니다.")

    curve = []
    for h in range(24):
        X = _build_vector(h, req.dow, req.month, req.year)
        curve.append(round(max(0.0, float(_STATE["model"].predict(X)[0])), 1))
    return {"시각별_예상세션수": curve}
