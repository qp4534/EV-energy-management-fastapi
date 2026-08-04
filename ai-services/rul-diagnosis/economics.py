# -*- coding: utf-8 -*-
"""
⑧ 경제성 · 환경 효과 산정

세 가지를 계산한다.
  1) 매각 수익      — 판매자(배터리 보유자) 관점
  2) 신품 대비 절감 — 구매자(재사용처) 관점: 신품 대신 이 배터리를 쓸 때 아끼는 돈
  3) 탄소 절감량    — 신품 제조를 회피해 줄이는 CO2e

⚠️ 정직성 고지
- 보유 데이터셋에 가격·탄소 정보가 없다. 아래 계수는 모두 **공개 문헌·시장 지표**이며,
  ML 학습 결과가 아니다. 결과는 참고용 추정치다.
- 탄소 절감은 '신품 제조 회피분'을 배터리의 남은 사용가능 기간에 귀속시키는
  일반적 회계 방식(avoided burden)을 따른다. 실제 LCA는 사용 단계 전력망
  탄소집약도·수송·재제조 공정까지 포함해야 하므로 값이 달라질 수 있다.

계수 근거
- 배터리 제조 탄소발자국 (셀 kWh당)
    NMC811 : 74 kg CO2e/kWh (중앙값; 5~95 백분위 59~115)  — Nature Comms 2024
    NMC622 : 56.1 kg CO2e/kWh (중앙값)                     — 동일 연구
    NMC111 : 64.3 kg CO2e/kWh (중앙값)                     — 동일 연구
    LFP    : 약 45 kg CO2e/kWh (NMC 대비 낮음, 보수적 적용)
- 재제조(remanufacturing) 공정 배출: 팩 생애 배출의 약 3% — iScience 2023
- 신품 배터리 팩 가격: BNEF 국제시세 약 $132/kWh ≈ 19만원/kWh
  (2026년 셀 기준 $100/kWh 이하 하락 전망 → 보수적으로 신품가 하단도 제공)
- 승용차 연간 배출 약 2.0 tCO2e, 소나무 1그루 연간 흡수 약 6.6 kg CO2e (환산용)
"""

# 화학조성별 제조 탄소발자국 (kg CO2e / kWh)
CHEMISTRY_CO2 = {
    "NMC811": {"median": 74.0, "low": 59.0, "high": 115.0,
               "label": "NMC811 (하이니켈)"},
    "NMC622": {"median": 56.1, "low": 45.0, "high": 90.0,
               "label": "NMC622"},
    "NMC111": {"median": 64.3, "low": 52.0, "high": 100.0,
               "label": "NMC111"},
    "LFP":    {"median": 45.0, "low": 35.0, "high": 70.0,
               "label": "LFP (리튬인산철)"},
}

REMANUFACTURE_RATIO = 0.03      # 재제조 공정 배출 = 신품 제조 배출의 약 3%
NEW_PACK_PRICE_KRW = 190_000    # 신품 팩 약 19만원/kWh (BNEF $132/kWh 환산)
NEW_PACK_PRICE_LOW = 145_000    # 2026년 하락 전망 반영 하단

# 환산 계수
CAR_ANNUAL_TCO2 = 2.0           # 승용차 1대 연간 배출 (tCO2e)
PINE_ANNUAL_KGCO2 = 6.6         # 소나무 1그루 연간 흡수 (kg CO2e)


# 등급별 SOH(잔존 용량) 추정 대역 — 업계 관행 기준
#   ⚠️ 우리 데이터엔 SOH가 없다. RUL 기반 '건강도'는 잔여 사이클 비율이지
#   잔존 용량(SOH)이 아니므로, 등급별 SOH 대역에 건강도를 매핑해 추정한다.
GRADE_SOH_BAND = {
    "1등급": (0.80, 0.95),   # SOH 80% 이상 — EV 계속 사용 가능
    "2등급": (0.60, 0.80),   # SOH 60~80% — ESS 등 저부하 재사용
    "3등급": (0.40, 0.60),   # SOH 60% 미만 — 재사용 기준 미달
}


def estimate_soh(grade: str, health_pct: float) -> float:
    """등급 + RUL 기반 건강도 -> 추정 SOH(0~1).

    등급이 SOH 대역을 정하고, 건강도가 그 대역 안에서의 위치를 정한다.
    """
    lo, hi = GRADE_SOH_BAND.get(grade, GRADE_SOH_BAND["3등급"])
    pos = max(0.0, min(1.0, float(health_pct) / 100.0))
    return lo + (hi - lo) * pos


def resolve_path(band_or_label: str | None, grade: str) -> str:
    """매도 경로 판정 — 'reuse' | 'ess' | 'material' | 'collect'.

    탄소 절감은 등급이 아니라 **이 배터리가 실제로 어디에 쓰이느냐**로 갈린다.
    같은 1등급 팩이라도 재제조업체에 팔면 신품 제조를 회피하지만,
    소재회수 업체에 팔면 회피하지 못한다. 경로가 없으면 등급 상한으로 대체한다.
    """
    if band_or_label:
        s = str(band_or_label)
        if s in ("reuse", "ess", "material", "collect"):
            return s
        if "중개" in s or "수거" in s:
            return "collect"
        if "재활용" in s or "소재" in s:
            return "material"
        if "2차사용" in s or "ESS" in s:
            return "ess"
        if "재사용" in s or "재제조" in s:
            return "reuse"
    return {"1등급": "reuse", "2등급": "ess"}.get(grade, "material")


def compute(capacity_kwh: float, sale_price_krw: float, grade: str,
            health_pct: float, chemistry: str = "NMC811",
            new_price_krw: float = NEW_PACK_PRICE_KRW,
            path: str | None = None) -> dict:
    """매각 수익·신품 대비 절감·탄소 절감을 함께 계산한다.

    Args:
        capacity_kwh: 팩 공칭 용량(kWh)
        sale_price_krw: ⑥에서 산정된 매각 예상가(원)
        grade: SOH 등급 (1등급 | 2등급 | 3등급)
        health_pct: RUL 기반 건강도(%) — 등급 SOH 대역 내 위치 결정에 사용
        chemistry: 화학조성 키
        new_price_krw: 신품 팩 단가(원/kWh)
        path: 매도 경로(band 키 또는 단가대 라벨). 생략 시 등급 상한으로 추정.
    """
    cap = float(capacity_kwh)
    chem = CHEMISTRY_CO2.get(chemistry, CHEMISTRY_CO2["NMC811"])
    soh = estimate_soh(grade, health_pct)
    usable = cap * soh                                       # 실사용 가능 용량(kWh)

    # ---- 1) 매각 수익 (판매자) ----
    revenue = float(sale_price_krw)

    # ---- 2) 신품 대비 절감 (구매자) ----
    # ⚠️ 이 비교는 매입처가 '신품 배터리 대신' 이 팩을 쓸 때만 성립한다.
    #    소재회수(재활용)·중개 업체는 배터리를 그대로 쓰는 게 아니므로
    #    신품 대비 절감이라는 개념 자체가 적용되지 않는다.
    route = resolve_path(path, grade)
    saving_ok = route in ("reuse", "ess")
    if saving_ok:
        # 재사용처가 같은 실사용 용량을 신품으로 확보하려면 드는 비용
        new_equiv_cost = usable * new_price_krw
        buyer_saving = max(0.0, new_equiv_cost - revenue)
        saving_rate = (buyer_saving / new_equiv_cost * 100) if new_equiv_cost > 0 else 0.0
        saving_note = ""
    else:
        new_equiv_cost = buyer_saving = saving_rate = 0.0
        saving_note = ("소재회수·중개 경로는 배터리를 그대로 사용하지 않으므로 "
                       "'신품 대비 절감'이 성립하지 않는다. 이 경로의 매입처 이득은 "
                       "회수 소재(리튬·니켈·코발트)의 시장가치에서 나온다.")

    # ---- 3) 탄소 절감 (CO2e) ----
    # 등급이 아니라 '실제 매도 경로'로 갈린다 — 1등급이어도 소재회수로 팔리면
    # 신품 제조를 회피하지 못한다.
    if route in ("material", "collect"):
        # 재활용: 신품 제조를 대체하지 못함. 소재 회수로 인한 절감분만 보수적으로 인정.
        # 회수 소재가 신품 제조 배출의 일부를 상쇄 (문헌상 약 20~30% 수준, 보수적 20%).
        co2_avoided = cap * chem["median"] * 0.20
        co2_lo = cap * chem["low"] * 0.20
        co2_hi = cap * chem["high"] * 0.20
        if route == "collect":
            co2_basis = ("수거·매입(중개) 경로 — 최종 용도가 정해지지 않아 재사용 "
                         "회피분을 인정할 수 없다. 소재회수 수준(신품 제조 배출의 "
                         "약 20% 상쇄)으로 보수적으로 계산")
        else:
            co2_basis = ("재활용 경로 — 신품 제조를 대체하지는 못하나, 회수 소재가 "
                         "신품 제조 배출의 약 20%를 상쇄하는 것으로 보수적으로 계산")
        reman = 0.0
    else:
        # 재사용/2차사용: 실사용 가능 용량만큼 신품 제조를 회피
        gross = usable * chem["median"]
        reman = gross * REMANUFACTURE_RATIO        # 재제조 공정에서 다시 발생하는 배출
        co2_avoided = max(0.0, gross - reman)
        co2_lo = max(0.0, usable * chem["low"] * (1 - REMANUFACTURE_RATIO))
        co2_hi = max(0.0, usable * chem["high"] * (1 - REMANUFACTURE_RATIO))
        co2_basis = (f"재사용 경로 — 실사용 가능 용량 {usable:.1f}kWh만큼 신품 제조를 회피. "
                     f"재제조 공정 배출(신품 제조의 {REMANUFACTURE_RATIO*100:.0f}%)은 차감")

    return {
        "용량_kWh": cap,
        "추정_SOH_퍼센트": round(soh * 100, 1),
        "실사용_kWh": round(usable, 2),
        "화학조성": chem["label"],
        # 경제성
        "매각수익_원": int(round(revenue)),
        "신품등가비용_원": int(round(new_equiv_cost)),
        "구매자절감_원": int(round(buyer_saving)),
        "절감률_퍼센트": round(saving_rate, 1),
        "절감_적용": saving_ok,
        "절감_미적용사유": saving_note,
        "신품단가_원per_kWh": int(new_price_krw),
        # 환경
        "탄소절감_kgCO2e": round(co2_avoided, 1),
        "탄소절감_범위_kgCO2e": (round(co2_lo, 1), round(co2_hi, 1)),
        "재제조배출_kgCO2e": round(reman, 1),
        "제조계수_kgCO2e_per_kWh": chem["median"],
        "탄소산정근거": co2_basis,
        "매도경로": route,
        # 환산
        "승용차_년_환산": round(co2_avoided / 1000 / CAR_ANNUAL_TCO2, 2),
        "소나무_그루_환산": int(round(co2_avoided / PINE_ANNUAL_KGCO2)),
    }


def co2_text(kg: float) -> str:
    """CO2e 표기 — kg/톤 자동 전환"""
    kg = float(kg)
    if abs(kg) >= 1000:
        return f"{kg/1000:.2f} tCO2e"
    return f"{kg:,.0f} kgCO2e"
