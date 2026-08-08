# -*- coding: utf-8 -*-
"""
⑥ 잔존가치 산정 · 매입처별 제안가 · 매도 제안서 생성

⚠️ 정직성 고지
- 보유 데이터셋에는 '가격' 정보가 없다. 가격은 ML 학습 결과가 아니라
  공개 실거래·시장 벤치마크에 ML 진단(등급·건전성)을 결합한 **산정값**이다.
- 아래 기업들의 '사업 영역'은 공개 자료로 확인된 사실이지만,
  **제안가는 해당 기업이 제시한 견적이 아니라 업종별 단가대에서 계산한 추정치**다.
  실제 거래는 반드시 개별 협의·실물 검사를 거쳐야 한다.

국내 시세 근거 (2023.3 공공입찰 및 시장 지표)
- 니로EV 64kWh 낙찰 785만원   → 122,656원/kWh
- 코나EV 64kWh 낙찰 1,150만원 → 179,688원/kWh  (둘 다 예정가의 3~4배 = 과열 피크)
- 공공 예정가(기준가) 역산     → 약 31,000~45,000원/kWh
- 신품 배터리 팩 국제시세 BNEF $132/kWh ≈ 약 19만원/kWh
- 2026년 신품 셀 $100/kWh 이하로 하락 → 2차사용 경제성 약화 추세
- 재활용 소재가치(파쇄 기준) £8~12/kWh ≈ 15,000~22,000원/kWh
"""

# ------------------------------------------------------------------
# 가격 산정 근거 출처 — 화면·PDF에 하이퍼링크로 노출한다.
# ------------------------------------------------------------------
PRICE_SOURCE_URL = (
    "https://about.bnef.com/insights/clean-energy/"
    "battery-pack-prices-fall-to-an-average-of-132-kwh-but-rising-commodity-prices-start-to-bite/"
)
PRICE_SOURCE_LABEL = "BloombergNEF 2021 리튬이온 배터리팩 가격조사 ($132/kWh)"

# ------------------------------------------------------------------
# 국내 시세 기준 단가대 (원/kWh)
#   하단 = 공공 예정가/침체 시황, 상단 = 2023 낙찰 피크
# ------------------------------------------------------------------
PRICE_BANDS = {
    "reuse": {   # EV 재제조·인증중고급
        "min": 80_000, "max": 180_000,
        "label": "재사용(EV 재제조)급",
        "basis": "국내 공공입찰 낙찰 122,656~179,688원/kWh(2023.3 피크) / "
                 "공공 예정가 약 31,000~45,000원/kWh가 하단. 신품 팩 시세 약 19만원/kWh 대비 40~85%.",
    },
    "ess": {     # 2차사용 ESS급
        "min": 40_000, "max": 90_000,
        "label": "2차사용(ESS)급",
        "basis": "해외 2차사용 재판매 £35~75/kWh 대역의 하단~중단, "
                 "국내 공공 예정가 상회 수준. 2026년 신품가 하락으로 상단 압박.",
    },
    "material": {  # 소재 회수급
        "min": 15_000, "max": 30_000,
        "label": "재활용(소재회수)급",
        "basis": "파쇄 후 유가금속 회수가치 £8~12/kWh. 코발트·니켈·리튬 시세 연동. "
                 "SOH와 무관하게 성립하는 최저 보장가 성격.",
    },
    "collect": {   # 수거·매입(중간 유통)급
        "min": 11_000, "max": 22_000,
        "label": "수거·매입(중개)급",
        "basis": "재활용사에 재판매하는 중간 유통 마진을 반영해 소재회수급의 약 75% 수준. "
                 "대신 전국 출장·방문수거로 소량·개별 처분이 간편.",
    },
}

# ------------------------------------------------------------------
# 실제 국내 매입처 (사업 영역은 공개 자료로 확인된 사실)
#   accepts : 매입 가능한 진단 등급
#   band    : 이 매입처가 지불하는 단가대
# ------------------------------------------------------------------
BUYERS = [
    # ---------------- 재사용(EV 재제조)급 ----------------
    {
        "name": "현대글로비스", "emoji": "🚛", "loc": "전국",
        "role": "폐배터리 회수·유통 (현대차그룹)",
        "band": "reuse", "accepts": ["1등급"],
        "why": "현대차그룹 순환경제 체계로 폐배터리를 수거하고 현대모비스가 재제조합니다. "
               "재제조 가능한 고SOH 팩에 가장 높은 값을 기대할 수 있는 경로입니다.",
        "fact": "현대글로비스 수거 → 현대모비스 재제조 순환경제 시스템 구축",
        "source_url": "https://www.hyundai.co.kr/story/CONT0000000000143438",
    },
    {
        "name": "지자체 공개입찰 (제주 배터리산업화센터 등)", "emoji": "🏛️", "loc": "제주 등",
        "role": "공공 사용후배터리 경쟁입찰",
        "band": "reuse", "accepts": ["1등급", "2등급"],
        "why": "경쟁입찰이라 시황이 좋으면 예정가의 3~4배까지 형성됩니다. "
               "단, 낙찰가 변동폭이 크고 입찰 일정이 정해져 있습니다.",
        "fact": "니로EV 64kWh 785만원 / 코나EV 64kWh 1,150만원 낙찰 (예정가의 3~4배)",
        # ⚠️ 위 낙찰 사례 수치는 자료 조사 당시 확보한 값으로, 재확인 가능한 1차 출처 링크를
        # 찾지 못했다(지어내지 않기 위해 비워둠). 아래 링크는 이 입찰을 관장하는 기관(제주
        # 전기차배터리산업화센터)의 공식 소개 페이지로, 절차·기관 신뢰성 참고용이다.
        "source_url": "https://battery.jejutp.or.kr/intro",
    },
    {
        "name": "에너지머티리얼즈 (GS건설 자회사)", "emoji": "🏗️", "loc": "경북 포항",
        "role": "재사용 + 재활용 통합",
        "band": "reuse", "accepts": ["1등급", "2등급"],
        "why": "수거·재사용과 블랙파우더 추출(재활용)을 함께 하는 곳이라, 등급이 애매하거나 "
               "혼합 물량이어도 한 곳에서 처리할 수 있습니다.",
        "fact": "리튬이온 배터리 수거·재사용 + 블랙파우더 추출, 연 2만톤 목표",
        "source_url": "https://dealsite.co.kr/articles/115154",
    },
    # ---------------- 2차사용(ESS)급 ----------------
    {
        "name": "피엠그로우", "emoji": "🔄", "loc": "경북 포항 블루밸리산단",
        "role": "재사용 전문 (ESS·UPS 전환)",
        "band": "ess", "accepts": ["1등급", "2등급"],
        "why": "국내 최초 재사용 전담 공장으로, SOH 기반 등급 분류 후 ESS·UPS로 재제조합니다. "
               "본 프로젝트의 '진단 → 등급 → 재사용처' 흐름과 가장 유사한 실제 사례입니다.",
        "fact": "국내 최초 재사용(Reuse) 전담 공장. SOH 기반 등급 분류 후 ESS·UPS 재제조",
        "source_url": "https://pmgrow.co.kr/about-us/introduction",
    },
    # ---------------- 재활용(소재회수)급 ----------------
    {
        "name": "성일하이텍", "emoji": "⚗️", "loc": "전북 군산 새만금",
        "role": "재활용 (전처리+후처리 통합)",
        "band": "material", "accepts": ["1등급", "2등급", "3등급"],
        "why": "국내 유일 전·후처리 통합 설비로 처리 규모가 가장 큽니다. "
               "재사용이 어려운 팩의 최종 처분처로 가장 확실합니다.",
        "fact": "국내 유일 전·후처리 통합, 전기차 약 30만대분 처리 규모",
        "source_url": "https://www.thelec.kr/news/articleView.html?idxno=28326",
    },
    {
        "name": "SK에코플랜트 (SK tes)", "emoji": "🏭", "loc": "경북 경주",
        "role": "재활용 (Ni·Co·Li 회수)",
        "band": "material", "accepts": ["1등급", "2등급", "3등급"],
        "why": "니켈·코발트·리튬을 자체 개발 기술로 회수합니다. 대기업 계열이라 "
               "대량·장기 계약 물량에 적합합니다.",
        "fact": "2025년 준공 목표 공장, 자체 개발 4대 기술 적용",
        "source_url": "https://news.skecoplant.com/sk-ecoplant/13958/",
    },
    {
        "name": "포스코HY클린메탈", "emoji": "🧪", "loc": "전남 여수 율촌산단",
        "role": "재활용 (블랙파우더 처리)",
        "band": "material", "accepts": ["1등급", "2등급", "3등급"],
        "why": "포스코·화유코발트·GS 합작사로 블랙파우더 처리 규모가 큽니다. "
               "남부권 물류에 유리합니다.",
        "fact": "포스코+화유코발트+GS 합작, 연 블랙파우더 1.2만톤",
        "source_url": "https://newsroom.posco.com/kr/%ED%8F%AC%EC%8A%A4%EC%BD%94%ED%99%80%EB%94%A9%EC%8A%A4-%EC%9D%B4%EC%B0%A8%EC%A0%84%EC%A7%80-%EB%A6%AC%EC%82%AC%EC%9D%B4%ED%81%B4%EB%A7%81-%EA%B3%B5%EC%9E%A5-%EC%A4%80%EA%B3%B5/",
    },
    {
        "name": "새빗켐", "emoji": "🔬", "loc": "경북 김천·상주",
        "role": "재활용 (전구체 복합액 생산)",
        "band": "material", "accepts": ["1등급", "2등급", "3등급"],
        "why": "회수 소재를 전구체 복합액으로 가공해 양극재 업계에 납품합니다. "
               "소재 밸류체인에 직접 연결된 처리사입니다.",
        "fact": "LG화학·한국전구체 등에 납품, 4개 공장 보유",
        "source_url": "https://www.thebell.co.kr/front/newsview.asp?key=202504231727562280102524",
    },
    {
        "name": "DS단석", "emoji": "🔋", "loc": "전북 군산",
        "role": "재활용",
        "band": "material", "accepts": ["1등급", "2등급", "3등급"],
        "why": "납축전지 리사이클 노하우를 리튬으로 확장한 곳입니다. "
               "중규모 물량 처리에 적합합니다.",
        "fact": "납축전지 리사이클 노하우 기반 확장, 연 8천톤 처리",
        "source_url": "https://www.thebell.co.kr/free/content/ArticleView.asp?key=202404091225487880106988",
    },
    {
        "name": "에스쓰리알 (S3R)", "emoji": "🧯", "loc": "충북 충주",
        "role": "재활용",
        "band": "material", "accepts": ["1등급", "2등급", "3등급"],
        "why": "50년 폐기물 재활용 노하우와 자체 R&D센터를 보유한 중견 처리사입니다. "
               "중부권 물류에 유리합니다.",
        "fact": "50년 폐기물 재활용 노하우, R&D센터 보유",
        "source_url": "https://s3r.co.kr/theme/daontheme_pro02/html/company/08.php",
    },
    {
        "name": "에코프로CnG", "emoji": "🌱", "loc": "—",
        "role": "폐배터리 재활용 (에코프로 자회사)",
        "band": "material", "accepts": ["1등급", "2등급", "3등급"],
        "why": "LG에너지솔루션과 장기 공급계약을 맺은 대형 재활용처로 물량 소화력이 큽니다. "
               "등급과 무관하게 매입됩니다.",
        "fact": "LG에너지솔루션과 2021년부터 4년간 폐배터리 장기공급계약 체결",
        "source_url": "https://www.etnews.com/20210208000138",
    },
    # ---------------- 수거·매입(중개)급 ----------------
    {
        "name": "디에이팩토리", "emoji": "🚚", "loc": "인천 서구 (전국 출장)",
        "role": "수거·매입 (전국 출장수거)",
        "band": "collect", "accepts": ["1등급", "2등급", "3등급"],
        "why": "정식 허가 수거업체로 전국 출장수거를 합니다. 중간 유통이라 단가는 낮지만, "
               "소량·개별 처분이나 즉시 처리에 가장 간편합니다.",
        "fact": "전기차·ESS·전동공구 배터리 정식 허가 수거업체",
        "source_url": "https://www.da-factory.co.kr/",
    },
    {
        "name": "GR리사이클", "emoji": "🚚", "loc": "전국 방문수거",
        "role": "수거·매입·재활용 (방문수거)",
        "band": "collect", "accepts": ["1등급", "2등급", "3등급"],
        "why": "환경부 정식 허가 업체로 전국 방문수거합니다. 소량 처분이나 긴급 처리에 적합합니다.",
        "fact": "환경부 정식 허가, 리튬이온 배터리(전기차·ESS·노트북) 전문",
        "source_url": "https://www.grrecycle.co.kr/",
    },
]

LEGAL_NOTE = ("『사용후배터리 산업 육성법』(2025.10 시행)에 따라 사용후 배터리는 "
              "지정된 회수·재활용 경로로만 반납해야 합니다. 임의 매각 전 적법 경로 여부를 "
              "반드시 확인하세요.")


# ------------------------------------------------------------------
# 단가대 위계 — 실제 용도는 '배터리가 감당 가능한 최고 용도'와
# '매입처가 수행하는 용도' 중 **낮은 쪽**으로 결정된다.
#   예) 2차사용 등급 배터리를 지자체 입찰(재제조급 매입처)에 팔아도
#       EV 재제조는 불가능하므로 실제 거래는 2차사용(ESS)급으로 성립.
# ------------------------------------------------------------------
BAND_RANK = {"collect": 0, "material": 1, "ess": 2, "reuse": 3}

# ⚠️ 색(공)은 'SOH 등급' 전용이다. 단가대(매도 경로)에는 등급 색을 쓰지 않는다.
#    같은 1등급 배터리도 어디에 파느냐에 따라 단가대가 달라지기 때문.
GRADE_INFO = {
    "1등급": {"dot": "🟢", "soh": "SOH 80% 이상",  "desc": "EV 계속 사용 가능"},
    "2등급": {"dot": "🟡", "soh": "SOH 60~80%",   "desc": "EV엔 부족, ESS 적합"},
    "3등급": {"dot": "🔴", "soh": "SOH 60% 미만", "desc": "재사용 기준 미달"},
}

GRADE_MAX_BAND = {
    "1등급": "reuse",     # SOH 80%↑ — EV 재제조까지 가능
    "2등급": "ess",       # SOH 60~80% — ESS 재사용까지만 가능
    "3등급": "material",  # SOH 60% 미만 — 소재 회수만 가능
}


def effective_band(buyer_band: str, grade: str) -> str:
    """매입처 단가대 + 배터리 등급 -> 실제 성립하는 단가대"""
    if buyer_band == "collect":
        return "collect"          # 중개상은 등급 무관 자체 단가대
    cap = GRADE_MAX_BAND.get(grade, "material")
    return buyer_band if BAND_RANK[buyer_band] <= BAND_RANK[cap] else cap


def _unit_price(band_key: str, condition: float) -> int:
    b = PRICE_BANDS[band_key]
    c = max(0.0, min(1.0, float(condition)))
    return int(round(b["min"] + (b["max"] - b["min"]) * c))


def estimate_offers(grade: str, capacity_kwh: float, condition: float,
                    spread: float = 0.15, buyers: list[dict] | None = None) -> list[dict]:
    """진단 등급·용량·상태로 매입처별 예상 제안가를 계산해 높은 순으로 반환.

    buyers를 생략하면 미리 조사해둔 고정 매입처 목록(BUYERS)을 쓴다. buyer_lookup.
    discover_buyers()로 실시간 검색해 찾은 매입처 목록을 넘기면, 회사만 그걸로 바꾸고
    가격 계산식(PRICE_BANDS - BNEF/국내 낙찰가 등으로 출처가 있는 벤치마크)은 그대로
    적용한다 - 가격 자체를 검색으로 지어내지 않기 위함."""
    buyers = buyers if buyers is not None else BUYERS
    out = []
    for b in buyers:
        if grade not in b["accepts"]:
            continue
        # 매입처 능력과 배터리 등급 중 낮은 쪽이 실제 용도(=단가대)
        band = effective_band(b["band"], grade)
        downgraded = BAND_RANK[band] < BAND_RANK[b["band"]]
        unit = _unit_price(band, condition)
        total = unit * float(capacity_kwh)
        out.append({
            "매입처": b["name"], "emoji": b["emoji"], "역할": b["role"],
            "위치": b.get("loc", "—"),
            "단가대": PRICE_BANDS[band]["label"],
            "요구_최소등급": {"reuse": "1등급", "ess": "2등급",
                          "material": "3등급", "collect": "등급 무관"}[band],
            "매입처_최고단가대": PRICE_BANDS[b["band"]]["label"],
            "등급제한_적용": downgraded,
            "단가_원per_kWh": unit,
            "제안가_원": int(round(total)),
            "제안가_범위_원": (int(round(total * (1 - spread))),
                          int(round(total * (1 + spread)))),
            "왜": b["why"], "확인된_사실": b["fact"],
            "단가근거": PRICE_BANDS[band]["basis"],
            # 근거 하이퍼링크 — 화면/PDF에서 "출처 보기" 링크로 노출한다.
            "출처_링크": b.get("source_url") or "",
            "단가출처_링크": PRICE_SOURCE_URL,
            "단가출처_라벨": PRICE_SOURCE_LABEL,
        })
    out.sort(key=lambda x: x["제안가_원"], reverse=True)
    return out


def won(n: float) -> str:
    """원화 표기 — 읽기 쉬운 만원/억원 단위"""
    n = float(n)
    if abs(n) >= 100_000_000:
        return f"{n/100_000_000:.2f}억원"
    if abs(n) >= 10_000:
        return f"{n/10_000:,.0f}만원"
    return f"{n:,.0f}원"


def build_offer_report(buyer: dict, *, capacity_kwh: float, grade: str,
                       rul_cycles: float, health_pct: float,
                       indicators: dict, full_life: float,
                       fire_note: str = "") -> str:
    """특정 매입처에게 제출할 **매도 제안서(사양서)** 마크다운 생성.

    이 배터리가 어떤 상태이고 왜 이 가격인지를 매입처가 판단할 수 있게 설명한다.
    """
    lo, hi = buyer["제안가_범위_원"]
    limit_note = ""
    if buyer.get("등급제한_적용"):
        limit_note = (f"\n> ※ 귀사는 {buyer['매입처_최고단가대']}까지 취급하시나, 본 배터리의 "
                      f"진단 등급이 **{grade}**이므로 {buyer['단가대']} 단가를 적용해 "
                      f"제안드립니다.\n")
    IND_KO = {"life": "수명 여유", "capacity": "방전 지속력",
              "charge": "충전 건전성", "stability": "전압 안정성"}
    ind_rows = "\n".join(
        f"| {IND_KO.get(k, k)} | {v*100:.0f} / 100 |" for k, v in indicators.items()
    )
    fire_block = f"\n### 안전성\n{fire_note}\n" if fire_note else ""

    return f"""# 사용후 배터리 매도 제안서

**수신** : {buyer['emoji']} **{buyer['매입처']}** ({buyer['역할']}) · 소재지 {buyer['위치']}
**건명** : 사용후 EV 배터리 팩 {capacity_kwh:g}kWh 매도 제안

---

## 1. 제안 가격

| 항목 | 내용 |
|---|---|
| **제안 단가** | **{buyer['단가_원per_kWh']:,}원 / kWh** ({buyer['단가대']}) |
| **제안 총액** | **{won(buyer['제안가_원'])}** |
| 협의 범위 | {won(lo)} ~ {won(hi)} |
| 용량 | {capacity_kwh:g} kWh |

> 단가 근거 — {buyer['단가근거']}
{limit_note}

## 2. 배터리 상태 진단 (AI 진단 결과)

| 지표 | 값 |
|---|---|
| **판별 등급** | **{grade}** |
| 예측 잔여 수명 | **{rul_cycles:,.0f} 사이클** (신품 기준 {full_life:,.0f} 사이클) |
| 추정 건강도 | **{health_pct:.1f}%** |

### 건전성 세부 지표 (0~100)

| 지표 | 점수 |
|---|---|
{ind_rows}

*진단 방식 — 충·방전 센서값을 RandomForest 회귀·분류 모델로 분석. 잔여수명 예측 평균오차 ±11 사이클, 등급 판별 정확도 98.4%.*
{fire_block}
## 3. 귀사에 적합한 이유

{buyer['왜']}

**확인된 사업 영역** — {buyer['확인된_사실']}

## 4. 유의사항

- 본 제안가는 공개 실거래·시장 벤치마크에 AI 진단 결과를 결합해 산정한 **추정치**이며, 귀사가 제시한 견적이 아닙니다.
- 최종 가격은 실물 검사(외관·전기적 검사)와 시황에 따라 조정될 수 있습니다.
- ⚖️ {LEGAL_NOTE}
"""
