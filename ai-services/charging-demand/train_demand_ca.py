# -*- coding: utf-8 -*-
"""
충전 수요 예측 모듈 (실측 데이터 버전)
데이터: EVChargingStationUsage.csv  (City of Palo Alto, 2011~2020, 259,415 세션, 47 스테이션)

- 세션들을 '도시 전체 시간당'으로 집계 -> 시간당 충전 세션 수(수요) 시계열
- 실제 도착 패턴(주간 피크·주말 저조·연도별 성장)이 살아있어 진짜 수요 예측이 됨
- 입력: 시각·요일·주말·월·연도  ->  출력: 그 시간의 예상 충전 세션 수
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.environ.get("CA_CSV", r"C:\Users\User\AppData\Local\Temp\EVChargingStationUsage.csv")

d = pd.read_csv(CSV, low_memory=False)
d.columns = [c.strip().lstrip("﻿") for c in d.columns]
d["ts"] = pd.to_datetime(d["Start Date"], errors="coerce")
d["energy"] = pd.to_numeric(d["Energy (kWh)"], errors="coerce")
d = d.dropna(subset=["ts"])
print("세션 수:", len(d), "| 기간:", d["ts"].min(), "~", d["ts"].max())

# ---- 시간당 집계 (도시 전체) ----
d["hour_ts"] = d["ts"].dt.floor("h")
hourly = d.groupby("hour_ts").agg(n_sessions=("ts", "size"),
                                  energy_kwh=("energy", "sum")).reset_index()
# 데이터가 있는 구간을 연속 시간 인덱스로 채워 '수요 0'인 시간도 포함
full = pd.date_range(hourly["hour_ts"].min(), hourly["hour_ts"].max(), freq="h")
hourly = hourly.set_index("hour_ts").reindex(full, fill_value=0).rename_axis("hour_ts").reset_index()

hourly["hour"] = hourly["hour_ts"].dt.hour
hourly["dow"] = hourly["hour_ts"].dt.dayofweek
hourly["is_weekend"] = hourly["dow"].isin([5, 6]).astype(int)
hourly["month"] = hourly["hour_ts"].dt.month
hourly["year"] = hourly["hour_ts"].dt.year
print("시간 슬롯 수:", len(hourly),
      "| 세션수 통계:", hourly["n_sessions"].describe().round(2).to_dict())

FEATURES = ["hour", "dow", "is_weekend", "month", "year"]
TARGET = "n_sessions"
X = hourly[FEATURES].values
y = hourly[TARGET].values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "RandomForest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                                  learning_rate=0.05, random_state=42),
}
results = {}
for name, m in models.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    results[name] = {"MAE": mean_absolute_error(yte, p),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, p))),
                     "R2": r2_score(yte, p)}
    print(f"[{name:16s}] MAE={results[name]['MAE']:.2f} "
          f"RMSE={results[name]['RMSE']:.2f} R2={results[name]['R2']:.4f}")

best_name = max(results, key=lambda k: results[k]["R2"])
best = models[best_name]
print(f"\n>>> Best: {best_name} (R2={results[best_name]['R2']:.4f})")

joblib.dump({"model": best, "features": FEATURES,
             "year_range": [int(hourly['year'].min()), int(hourly['year'].max())],
             "max_sessions": int(hourly['n_sessions'].max())},
            os.path.join(HERE, "demand_ca_model.joblib"))
with open(os.path.join(HERE, "demand_ca_metrics.json"), "w", encoding="utf-8") as f:
    json.dump({"results": results, "best_model": best_name}, f, indent=2, ensure_ascii=False)

# 시각별 평균 수요 곡선 (실측) 저장 → 그래프
plt.figure(figsize=(7, 4))
hourly.groupby("hour")["n_sessions"].mean().plot(marker="o")
plt.xlabel("Hour of day"); plt.ylabel("Avg sessions / hour")
plt.title("Palo Alto — Real Hourly Charging Demand")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(HERE, "demand_ca_hourly.png"), dpi=120); plt.close()

print("\n산출물: demand_ca_model.joblib / demand_ca_metrics.json / demand_ca_hourly.png")
