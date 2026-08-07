# -*- coding: utf-8 -*-
"""매도 제안서 PDF 생성 — 완성차 기업 공식문서 스타일

디자인 기준
- 상단 네이비 헤더 바 + 문서 식별정보(문서번호/작성일/작성부서)
- 번호 매긴 조항식 섹션 (1. 제안 가격 / 2. 배터리 상태 진단 / …)
- 표 헤더 네이비, 본문 흰 배경 + 얇은 회색 괘선
- 하단 고정 푸터(문서명·페이지·기밀표시)
- 한글: 맑은 고딕(Regular/Bold) 임베드
"""
import os
import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, KeepTogether)

# ---------------- 브랜드 색상 ----------------
NAVY = colors.HexColor("#002C5F")      # 기업 시그니처 네이비
NAVY_LT = colors.HexColor("#0B4A8F")
GRAY_BG = colors.HexColor("#F4F6F8")
GRAY_LINE = colors.HexColor("#D5DBE1")
GRAY_TXT = colors.HexColor("#5A6672")
ACCENT = colors.HexColor("#00AAD2")

HERE = os.path.dirname(os.path.abspath(__file__))

FONT_R, FONT_B = "MalgunGothic", "MalgunGothic-Bold"
_FONTS_READY = False


def _register_fonts():
    """한글 폰트 등록.

    배포 컨테이너(Linux)에는 맑은 고딕이 없으므로, 이 디렉터리에 함께 담아 배포하는
    번들 폰트(fonts/NotoSansKR-*.ttf, OFL 라이선스)를 우선 쓴다. 번들 폰트가 없는
    환경(예: 이 리포를 원본 그대로 로컬 윈도우에서 돌릴 때)에서는 시스템 맑은 고딕으로
    폴백한다.
    """
    global _FONTS_READY
    if _FONTS_READY:
        return

    bundled_reg = os.path.join(HERE, "fonts", "NotoSansKR-Regular.ttf")
    bundled_bold = os.path.join(HERE, "fonts", "NotoSansKR-Bold.ttf")

    if os.path.exists(bundled_reg):
        reg = bundled_reg
        bold = bundled_bold if os.path.exists(bundled_bold) else bundled_reg
    else:
        win = os.environ.get("WINDIR", r"C:\Windows")
        reg = os.path.join(win, "Fonts", "malgun.ttf")
        bold_candidate = os.path.join(win, "Fonts", "malgunbd.ttf")
        bold = bold_candidate if os.path.exists(bold_candidate) else reg
        if not os.path.exists(reg):
            raise RuntimeError(
                "한글 폰트를 찾을 수 없습니다. fonts/NotoSansKR-Regular.ttf를 이 "
                "디렉터리에 두거나, 윈도우 환경에서 맑은 고딕을 사용하세요."
            )

    pdfmetrics.registerFont(TTFont(FONT_R, reg))
    pdfmetrics.registerFont(TTFont(FONT_B, bold))
    _FONTS_READY = True


def _styles():
    return {
        "h1": ParagraphStyle("h1", fontName=FONT_B, fontSize=19, leading=25,
                             textColor=colors.white),
        "sub": ParagraphStyle("sub", fontName=FONT_R, fontSize=9.5, leading=14,
                              textColor=colors.HexColor("#B9CBE0")),
        "sec": ParagraphStyle("sec", fontName=FONT_B, fontSize=12, leading=17,
                              textColor=NAVY, spaceBefore=2, spaceAfter=6),
        "body": ParagraphStyle("body", fontName=FONT_R, fontSize=9.5, leading=15,
                               textColor=colors.HexColor("#1E2A35")),
        "small": ParagraphStyle("small", fontName=FONT_R, fontSize=8.3, leading=12.5,
                                textColor=GRAY_TXT),
        "cell": ParagraphStyle("cell", fontName=FONT_R, fontSize=9, leading=13.5,
                               textColor=colors.HexColor("#1E2A35")),
        "cellb": ParagraphStyle("cellb", fontName=FONT_B, fontSize=9, leading=13.5,
                                textColor=NAVY),
        "big": ParagraphStyle("big", fontName=FONT_B, fontSize=15, leading=20,
                              textColor=NAVY),
        "note": ParagraphStyle("note", fontName=FONT_R, fontSize=8.3, leading=13,
                               textColor=colors.HexColor("#8A3A1E")),
    }


def _kv_table(rows, widths, st, header=None):
    """라벨-값 표. header가 있으면 네이비 헤더행 추가."""
    data = []
    if header:
        data.append([Paragraph(f"<b>{h}</b>", ParagraphStyle(
            "th", fontName=FONT_B, fontSize=9, leading=13,
            textColor=colors.white)) for h in header])
    for r in rows:
        data.append([Paragraph(str(c), st["cell"]) if i else
                     Paragraph(str(c), st["cellb"]) for i, c in enumerate(r)])

    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, GRAY_LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]
    start = 0
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.6, NAVY)]
        start = 1
    style.append(("BACKGROUND", (0, start), (0, -1), GRAY_BG))
    t.setStyle(TableStyle(style))
    return t


def build_pdf(*, buyer: dict, capacity_kwh: float, grade: str,
              rul_cycles: float, health_pct: float, indicators: dict,
              full_life: float, won, seller: str = "배터리 진단 AI 시스템",
              doc_no: str | None = None, eco: dict | None = None,
              fire_note: str = "") -> bytes:
    """매도 제안서 PDF 바이트 반환."""
    _register_fonts()
    st = _styles()
    today = date.today()
    doc_no = doc_no or f"BAT-{today:%Y%m%d}-{abs(hash(buyer['매입처'])) % 10000:04d}"

    buf = io.BytesIO()
    PW, PH = A4
    ML, MR = 18 * mm, 18 * mm
    HEADER_H = 34 * mm

    def page(canv, _doc):
        canv.saveState()
        # 상단 네이비 헤더
        canv.setFillColor(NAVY)
        canv.rect(0, PH - HEADER_H, PW, HEADER_H, stroke=0, fill=1)

        canv.setFillColor(colors.white)
        canv.setFont(FONT_B, 17)
        canv.drawString(ML, PH - 16 * mm, "사용후 배터리 매도 제안서")
        canv.setFont(FONT_R, 8.6)
        canv.setFillColor(colors.HexColor("#B9CBE0"))
        canv.drawString(ML, PH - 22.5 * mm,
                        "Second-Life EV Battery Sales Proposal")
        canv.setFont(FONT_R, 8.2)
        canv.drawRightString(PW - MR, PH - 14 * mm, f"문서번호  {doc_no}")
        canv.drawRightString(PW - MR, PH - 19 * mm, f"작성일자  {today:%Y-%m-%d}")
        canv.drawRightString(PW - MR, PH - 24 * mm, f"작 성 자  {seller}")

        # 푸터
        canv.setStrokeColor(GRAY_LINE)
        canv.setLineWidth(0.5)
        canv.line(ML, 14 * mm, PW - MR, 14 * mm)
        canv.setFont(FONT_R, 7.6)
        canv.setFillColor(GRAY_TXT)
        canv.drawString(ML, 9.5 * mm, "사용후 배터리 매도 제안서 · 대외비(Confidential)")
        canv.drawRightString(PW - MR, 9.5 * mm, f"- {canv.getPageNumber()} -")
        canv.restoreState()

    doc = BaseDocTemplate(buf, pagesize=A4,
                          leftMargin=ML, rightMargin=MR,
                          topMargin=HEADER_H + 8 * mm, bottomMargin=20 * mm,
                          title="사용후 배터리 매도 제안서", author=seller)
    frame = Frame(ML, 20 * mm, PW - ML - MR, PH - HEADER_H - 28 * mm, id="f")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=page)])

    CW = PW - ML - MR
    S = []

    # ── 수신 정보
    S.append(_kv_table([
        ["수 신", f"<b>{buyer['매입처']}</b>  ({buyer['역할']})"],
        ["소재지", buyer.get("위치", "—")],
        ["건 명", f"사용후 EV 배터리 팩 {capacity_kwh:g}kWh 매도 제안"],
    ], [28 * mm, CW - 28 * mm], st))
    S.append(Spacer(1, 9 * mm))

    # ── 1. 제안 가격
    lo, hi = buyer["제안가_범위_원"]
    S.append(Paragraph("1. 제안 가격", st["sec"]))
    price_tbl = Table([[
        Paragraph("제안 총액", ParagraphStyle("pl", fontName=FONT_R, fontSize=9,
                                          textColor=colors.HexColor("#B9CBE0"))),
        Paragraph(f"<b>{won(buyer['제안가_원'])}</b>",
                  ParagraphStyle("pv", fontName=FONT_B, fontSize=20, leading=25,
                                 textColor=colors.white, alignment=TA_RIGHT)),
    ]], colWidths=[CW * 0.45, CW * 0.55])
    price_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    S.append(price_tbl)
    S.append(Spacer(1, 2 * mm))
    S.append(_kv_table([
        ["적용 단가", f"{buyer['단가_원per_kWh']:,} 원 / kWh　({buyer['단가대']})"],
        ["협의 범위", f"{won(lo)}　~　{won(hi)}"],
        ["공칭 용량", f"{capacity_kwh:g} kWh"],
    ], [28 * mm, CW - 28 * mm], st))
    S.append(Spacer(1, 3 * mm))
    S.append(Paragraph(f"※ 단가 근거 — {buyer['단가근거']}", st["small"]))
    if buyer.get("등급제한_적용"):
        S.append(Spacer(1, 1.5 * mm))
        S.append(Paragraph(
            f"※ 귀사는 {buyer['매입처_최고단가대']}까지 취급하시나, 본 배터리의 진단 등급이 "
            f"'{grade}'이므로 {buyer['단가대']} 단가를 적용하여 제안드립니다.", st["note"]))
    S.append(Spacer(1, 8 * mm))

    # ── 2. 배터리 상태 진단
    S.append(Paragraph("2. 배터리 상태 진단 (AI 진단 결과)", st["sec"]))
    S.append(_kv_table([
        ["판별 등급", f"<b>{grade}</b>"],
        ["예측 잔여수명", f"<b>{rul_cycles:,.0f} 사이클</b>　(신품 기준 {full_life:,.0f} 사이클)"],
        ["추정 건강도", f"<b>{health_pct:.1f} %</b>"],
    ], [32 * mm, CW - 32 * mm], st))
    S.append(Spacer(1, 4 * mm))

    IND_KO = {"life": "수명 여유", "capacity": "방전 지속력",
              "charge": "충전 건전성", "stability": "전압 안정성"}
    ind_rows = [[IND_KO.get(k, k), f"{v*100:.0f} / 100"] for k, v in indicators.items()]
    S.append(_kv_table(ind_rows, [CW * 0.5, CW * 0.5], st,
                       header=["건전성 세부 지표", "점수"]))
    S.append(Spacer(1, 3 * mm))
    S.append(Paragraph(
        "진단 방식 — 충·방전 센서값을 RandomForest 회귀·분류 모델로 분석. "
        "잔여수명 예측 평균오차 ±11 사이클, 등급 판별 정확도 98.4%.", st["small"]))
    S.append(Spacer(1, 4 * mm))

    # 등급 판정 기준 — SOH(추정 건강도) + 전압 안정성 지표를 함께 보고
    # 매도 경로 상한(재사용/2차사용/재활용)을 정하는 기준을 명시한다.
    # KeepTogether로 묶어 표가 페이지 경계에서 잘리지 않게 한다.
    S.append(KeepTogether([
        Paragraph("등급 판정 기준", st["cellb"]),
        Spacer(1, 1.5 * mm),
        _kv_table([
            ["SOH 80% 이상 + 전압 안정성 우수", "재사용(EV 재제조)급"],
            ["SOH 60~80% + 안정성 확보", "2차사용(ESS)급"],
            ["SOH 60% 미만 또는 이상 징후", "재활용(소재회수)급"],
        ], [CW * 0.62, CW * 0.38], st, header=["평가 결과", "판정"]),
        Spacer(1, 2 * mm),
        Paragraph(
            "SOH는 데이터에 없어 등급별 SOH 대역(1등급 80~100%, 2등급 60~80%)에 진단 "
            "건강도를 매핑해 추정합니다. 3등급(재활용)은 하한을 두지 않았습니다 — 재활용은 "
            "배터리를 분해·용해해 원재료를 추출하는 공정이라 SOH 성능과 무관하게 처리 가능하며, "
            "실제 문헌에도 재활용에 성능 하한을 두는 사례가 없습니다. 전압 안정성은 위 건전성 "
            "세부 지표의 '전압 안정성' 점수를 기준으로 판단합니다.", st["small"]),
        Spacer(1, 1.5 * mm),
        Paragraph(
            "판정 기준 출처 — SOH 60%·80% 컷오프는 이차전지 재사용 업계 자료(ROPLANT: 80~90%"
            "이상 재제조 / 60~80% 재사용 / 60%미만 재활용)와 한국에너지경제연구원(KESRC) 정책연구"
            "(완성차 성능보증 통상 70~80%, ESS 재사용 시 초기용량 60%까지 사용 후 폐기 가능)를 "
            "함께 따랐습니다. ※ 문헌에 따라 '재사용'·'재제조' 명칭이 반대로 쓰이는 경우가 있습니다"
            "(예: 한국소방안전원(KFPA) 자료는 80% 이상을 '재사용', 65~80%를 '재제조'로 표기) — "
            "본 제안서는 '차량 재장착이 타 용도 전용보다 더 높은 건강 상태를 요구한다'는 실무 "
            "논리에 따라 고SOH 구간을 재제조(EV 재장착)급으로 정의했습니다. 정부는 전기차 배터리 "
            "탈거 전 성능평가를 거쳐 재제조·재사용 가능 배터리를 '순환자원'으로 지정하는 제도를 "
            "추진 중이며, 본 등급 체계는 이 정책 방향과 연동 가능하도록 설계했습니다.", st["small"]),
        Spacer(1, 1.5 * mm),
        Paragraph(
            "재활용 등급 내 이상징후 취급 — 재활용은 SOH 하한이 없는 대신, 물리적 손상·열폭주 "
            "전조 같은 안전 이상은 별도로 감지합니다. 본 시스템의 화재 위험 게이트(Agent1, "
            "fire_risk_model)가 위험을 감지하면 SOH 등급과 무관하게 즉시 폐기·특별취급 대상으로 "
            "분류하고 후속 등급 판정을 진행하지 않습니다. 실무 문헌은 내부저항(IR) 증가율·셀간 "
            "전압편차 등도 함께 볼 것을 권장하나, 본 모델은 팩 단위 충·방전 센서값만 확보돼 있어 "
            "전압 강하 패턴 기반 지표로 안정성을 근사합니다 — 셀 단위 IR·전압편차 계측은 후속 "
            "데이터 확보 과제입니다.", st["small"]),
    ]))
    if fire_note:
        S.append(Spacer(1, 2 * mm))
        S.append(Paragraph(f"안전성 — {fire_note}", st["body"]))
    S.append(Spacer(1, 8 * mm))

    # ── 3. 경제성·환경 효과 (선택)
    if eco:
        S.append(Paragraph("3. 경제성 · 환경 효과", st["sec"]))
        # 신품 대체가 성립하는 경로(재사용·2차사용)에서만 절감액을 제시한다
        if eco.get("절감_적용", True):
            _cost_rows = [
                ["구매자 절감액", f"<b>{won(eco['구매자절감_원'])}</b>　"
                              f"(신품 대비 {eco['절감률_퍼센트']:.1f}% 절감)"],
                ["신품 등가 비용", won(eco["신품등가비용_원"])],
            ]
        else:
            _cost_rows = [["신품 대비 절감", "해당 없음 — " + eco.get("절감_미적용사유", "")]]
        # 3등급(재활용)은 '실사용 kWh'도 SOH 하한 없는 값에서 파생된 수치라
        # 같이 감춘다 — 애초에 재활용 CO2 계산에도 이 값을 쓰지 않는다(용량 무관 20% 고정).
        _soh_cell = ("60% 미만　(재활용은 용량 무관 처리 — 실사용 kWh 산정 불필요)"
                    if eco.get("SOH_하한_근거없음")
                    else f"{eco['추정_SOH_퍼센트']} %　(실사용 {eco['실사용_kWh']} kWh)")
        S.append(_kv_table(_cost_rows + [
            ["추정 SOH", _soh_cell],
            ["CO₂ 절감량", f"<b>{eco['탄소절감_kgCO2e']:,.0f} kgCO₂e</b>　"
                        f"(승용차 {eco['승용차_년_환산']:.2f}년 배출 / 소나무 "
                        f"{eco['소나무_그루_환산']:,}그루 연간 흡수량)"],
        ], [32 * mm, CW - 32 * mm], st))
        S.append(Spacer(1, 3 * mm))
        S.append(Paragraph(
            "탄소 계수 — 제조 탄소발자국 Nature Communications(2024), "
            "재제조 공정 배출 3% iScience(2023) 적용.", st["small"]))
        S.append(Spacer(1, 8 * mm))
        n4, n5 = "4", "5"
    else:
        n4, n5 = "3", "4"

    # ── 4. 귀사 적합성
    S.append(KeepTogether([
        Paragraph(f"{n4}. 귀사 적합성", st["sec"]),
        Paragraph(buyer["왜"], st["body"]),
        Spacer(1, 2 * mm),
        Paragraph(f"확인된 사업 영역 — {buyer['확인된_사실']}", st["small"]),
    ]))
    S.append(Spacer(1, 8 * mm))

    # ── 5. 유의사항
    S.append(Paragraph(f"{n5}. 유의사항", st["sec"]))
    for txt in [
        "본 제안가는 공개 실거래·시장 벤치마크에 AI 진단 결과를 결합하여 산정한 추정치이며, "
        "귀사가 제시한 견적이 아닙니다.",
        "최종 가격은 실물 검사(외관·전기적 검사) 및 시황에 따라 조정될 수 있습니다.",
        "『사용후배터리 산업 육성법』(2025.10 시행)에 따라 사용후 배터리는 지정된 회수·재활용 "
        "경로로만 반납해야 하며, 매각 전 적법 경로 여부를 확인하여야 합니다.",
    ]:
        S.append(Paragraph(f"·　{txt}", st["small"]))
        S.append(Spacer(1, 1.5 * mm))

    # ── 서명란
    S.append(Spacer(1, 10 * mm))
    sign = Table([["제 안 자", seller, "( 인 )"]],
                 colWidths=[26 * mm, CW - 26 * mm - 26 * mm, 26 * mm])
    sign.setStyle(TableStyle([
        ("FONT", (0, 0), (0, 0), FONT_B, 9.5),
        ("FONT", (1, 0), (-1, 0), FONT_R, 9.5),
        ("TEXTCOLOR", (0, 0), (0, 0), NAVY),
        ("ALIGN", (2, 0), (2, 0), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, NAVY),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    S.append(sign)

    doc.build(S)
    return buf.getvalue()


def build_pdf_from_view(*, buyer_name: str, buyer_role: str = "", buyer_location: str = "",
                        price_total_manwon, unit_price_won, negotiation_range: str,
                        price_grade_label: str, price_note: str = "",
                        grade: str, remaining_cycle, new_cycle,
                        health_score_pct, health_metrics: list[dict],
                        diagnosis_note: str = "", reasons: list[str] | None = None,
                        cautions: list[str] | None = None,
                        seller: str = "배터리 진단 AI 시스템",
                        doc_no: str | None = None) -> bytes:
    """관리자 웹(BatteryDiagnosis.jsx "배터리 매도 제안서" 탭)에 이미 표시된 값을 그대로
    받아 PDF로 렌더링한다.

    build_pdf()와 달리 원본 센서값으로 Agent1~3을 다시 돌리지 않는다 - 화면에 보이는
    숫자(BATTERY_PASSPORT/BATTERY_PROPOSALS/BATTERY_DIAGNOSIS_METRICS에서 이미 계산되어
    저장된 값)를 그대로 문서화하는 용도라서, 화면과 PDF 숫자가 어긋날 일이 없다.
    health_metrics는 [{"label": "수명 여유", "score": "50 / 100"}, ...] 형태(이미
    포맷된 문자열)를 그대로 받는다.
    """
    _register_fonts()
    st = _styles()
    today = date.today()
    doc_no = doc_no or f"BAT-{today:%Y%m%d}-{abs(hash(buyer_name)) % 10000:04d}"
    reasons = reasons or []
    cautions = cautions or []

    buf = io.BytesIO()
    PW, PH = A4
    ML, MR = 18 * mm, 18 * mm
    HEADER_H = 34 * mm

    def page(canv, _doc):
        canv.saveState()
        canv.setFillColor(NAVY)
        canv.rect(0, PH - HEADER_H, PW, HEADER_H, stroke=0, fill=1)

        canv.setFillColor(colors.white)
        canv.setFont(FONT_B, 17)
        canv.drawString(ML, PH - 16 * mm, "사용후 배터리 매도 제안서")
        canv.setFont(FONT_R, 8.6)
        canv.setFillColor(colors.HexColor("#B9CBE0"))
        canv.drawString(ML, PH - 22.5 * mm, "Second-Life EV Battery Sales Proposal")
        canv.setFont(FONT_R, 8.2)
        canv.drawRightString(PW - MR, PH - 14 * mm, f"문서번호  {doc_no}")
        canv.drawRightString(PW - MR, PH - 19 * mm, f"작성일자  {today:%Y-%m-%d}")
        canv.drawRightString(PW - MR, PH - 24 * mm, f"작 성 자  {seller}")

        canv.setStrokeColor(GRAY_LINE)
        canv.setLineWidth(0.5)
        canv.line(ML, 14 * mm, PW - MR, 14 * mm)
        canv.setFont(FONT_R, 7.6)
        canv.setFillColor(GRAY_TXT)
        canv.drawString(ML, 9.5 * mm, "사용후 배터리 매도 제안서 · 대외비(Confidential)")
        canv.drawRightString(PW - MR, 9.5 * mm, f"- {canv.getPageNumber()} -")
        canv.restoreState()

    doc = BaseDocTemplate(buf, pagesize=A4,
                          leftMargin=ML, rightMargin=MR,
                          topMargin=HEADER_H + 8 * mm, bottomMargin=20 * mm,
                          title="사용후 배터리 매도 제안서", author=seller)
    frame = Frame(ML, 20 * mm, PW - ML - MR, PH - HEADER_H - 28 * mm, id="f")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=page)])

    CW = PW - ML - MR
    S = []

    S.append(_kv_table([
        ["수 신", f"<b>{buyer_name}</b>" + (f"  ({buyer_role})" if buyer_role else "")],
        ["소재지", buyer_location or "—"],
    ], [28 * mm, CW - 28 * mm], st))
    S.append(Spacer(1, 9 * mm))

    S.append(Paragraph("1. 제안 가격", st["sec"]))
    price_tbl = Table([[
        Paragraph("제안 총액", ParagraphStyle("pl", fontName=FONT_R, fontSize=9,
                                          textColor=colors.HexColor("#B9CBE0"))),
        Paragraph(f"<b>{price_total_manwon:,.0f}만원</b>",
                  ParagraphStyle("pv", fontName=FONT_B, fontSize=20, leading=25,
                                 textColor=colors.white, alignment=TA_RIGHT)),
    ]], colWidths=[CW * 0.45, CW * 0.55])
    price_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    S.append(price_tbl)
    S.append(Spacer(1, 2 * mm))
    S.append(_kv_table([
        ["적용 단가", f"{unit_price_won:,.0f} 원 / kWh　({price_grade_label})"],
        ["협의 범위", negotiation_range],
    ], [28 * mm, CW - 28 * mm], st))
    if price_note:
        S.append(Spacer(1, 3 * mm))
        S.append(Paragraph(f"※ {price_note}", st["small"]))
    S.append(Spacer(1, 8 * mm))

    S.append(Paragraph("2. 배터리 상태 진단 (AI 진단 결과)", st["sec"]))
    S.append(_kv_table([
        ["판별 등급", f"<b>{grade}</b>"],
        ["예측 잔여수명", f"<b>{remaining_cycle:,.0f}</b> 사이클　(신품 기준 {new_cycle:,.0f} 사이클)"],
        ["추정 건강도", f"<b>{health_score_pct:.1f}%</b>"],
    ], [32 * mm, CW - 32 * mm], st))
    S.append(Spacer(1, 4 * mm))

    if health_metrics:
        S.append(_kv_table(
            [[m.get("label", ""), m.get("score", "")] for m in health_metrics],
            [CW * 0.5, CW * 0.5], st, header=["건전성 세부 지표", "점수"]))
        S.append(Spacer(1, 3 * mm))
    if diagnosis_note:
        S.append(Paragraph(diagnosis_note, st["small"]))
    S.append(Spacer(1, 8 * mm))

    if reasons:
        S.append(Paragraph("3. 귀사에 적합한 이유", st["sec"]))
        for r in reasons:
            S.append(Paragraph(r, st["body"]))
        S.append(Spacer(1, 8 * mm))

    if cautions:
        S.append(Paragraph("4. 유의사항", st["sec"]))
        for c in cautions:
            S.append(Paragraph(f"·　{c}", st["small"]))
            S.append(Spacer(1, 1.5 * mm))

    S.append(Spacer(1, 10 * mm))
    sign = Table([["제 안 자", seller, "( 인 )"]],
                 colWidths=[26 * mm, CW - 26 * mm - 26 * mm, 26 * mm])
    sign.setStyle(TableStyle([
        ("FONT", (0, 0), (0, 0), FONT_B, 9.5),
        ("FONT", (1, 0), (-1, 0), FONT_R, 9.5),
        ("TEXTCOLOR", (0, 0), (0, 0), NAVY),
        ("ALIGN", (2, 0), (2, 0), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, NAVY),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    S.append(sign)

    doc.build(S)
    return buf.getvalue()
