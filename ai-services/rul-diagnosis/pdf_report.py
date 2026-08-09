# -*- coding: utf-8 -*-
"""매도 제안서 PDF 생성 — 정부 부처 보도자료 스타일

디자인 기준
- 상단: "MijungE" 로고 lockup + 헤드라인 타이틀 + 문서정보 표(우측, 담당자 연락처 포함).
  강조는 상단 색상 바 없이 로고·타이포그래피만으로 처리한다.
- 표 헤더: 옅은 회색(#EAEAEA) 배경 + 검정 굵은 글씨 (네이비 배경 대신 - 강조는 폰트로만)
- 섹션 제목: "1. 2. 3. …" 순수 숫자 목록 (원문자·박스형 넘버링은 쓰지 않는다)
- 문체: 개조식 명사형 종결("~함"/"~임"), 긴 문장은 짧게 분리, 대시(—) 연결과 화살표(→)
  대신 마침표/콜론/쉼표. □(대분류)/-(부연, 대시) 불릿 사용. 화면에서 넘어오는 자유 목록
  (extra_reasons 등)은 □/-와 헷갈리지 않도록 번호("1. 2. 3.") 목록으로 구분한다.
- 전문용어(SOH·RUL·RandomForest 등)는 본문에 별표(*)만 달고, 문서 맨 끝 "용어 설명"
  섹션에서 한 번에 정의한다. 괄호 안 세부조건·부가정보는 본문 문장 대신 바로 아래
  "-" 하위 항목으로 분리해 문장을 짧게 유지한다.
- 하단 고정 푸터(문서명·페이지·기밀표시)
- 한글: 맑은 고딕(Regular/Bold) 또는 번들 Noto Sans KR 임베드

⚠️ 용어: "사용후 배터리"를 상위 카테고리 명칭으로 쓴다(현대자동차그룹 공식 스토리 페이지 기준 -
https://www.hyundaimotorgroup.com/ko/story/CONT0000000000143438 - 업계에서도 "사용후 배터리"를
전체 범주로 쓰고, 그 안에서 재사용/2차사용/재활용으로 세분화하는 게 표준 용어다). 표 안의 등급
명칭("재사용(EV 재제조)급"/"2차사용(ESS)급"/"재활용(소재회수)급")은 세부 분류 용어라 손대지
않았고, 유의사항의 실제 법령명(「사용후 배터리의 관리 및 산업육성에 관한 법률」)도 그대로 유지한다.
"""
import os
import re
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
NAVY = colors.HexColor("#002C5F")      # 포인트 색(로고/구분선/강조 숫자 박스 전용 - 배경 채움엔 안 씀)
NAVY_LT = colors.HexColor("#0B4A8F")
GRAY_BG = colors.HexColor("#F4F6F8")
GRAY_HEADER = colors.HexColor("#EAEAEA")  # 표 헤더 배경(정부 보도자료 스타일)
GRAY_LINE = colors.HexColor("#D5DBE1")
GRAY_TXT = colors.HexColor("#5A6672")
INK = colors.HexColor("#1A1A1A")          # 표 헤더/헤드라인용 검정에 가까운 잉크색
# 강조 색(하늘색/시안색 계열)은 쓰지 않는다 - 정부 보도자료·기업 공문 톤에 맞춰 링크·
# 헤더 모두 무채색 계열로 통일한다.

# ---------------- 담당자 정보 (헤더 문서정보 표에 노출) ----------------
CONTACT_NAME = "김진단"
CONTACT_EMAIL = "tkyaho@mijungev.kro.kr"

HERE = os.path.dirname(os.path.abspath(__file__))

FONT_R, FONT_B = "MalgunGothic", "MalgunGothic-Bold"
_FONTS_READY = False

_CIRCLED = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥", 7: "⑦"}


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
        "sec": ParagraphStyle("sec", fontName=FONT_B, fontSize=12, leading=15,
                              textColor=INK, spaceBefore=0, spaceAfter=0),
        "body": ParagraphStyle("body", fontName=FONT_R, fontSize=9.5, leading=15,
                               textColor=colors.HexColor("#1E2A35")),
        "small": ParagraphStyle("small", fontName=FONT_R, fontSize=8.3, leading=12.5,
                                textColor=GRAY_TXT),
        "cell": ParagraphStyle("cell", fontName=FONT_R, fontSize=9, leading=13.5,
                               textColor=colors.HexColor("#1E2A35")),
        "cellb": ParagraphStyle("cellb", fontName=FONT_B, fontSize=9, leading=13.5,
                                textColor=INK),
        "note": ParagraphStyle("note", fontName=FONT_R, fontSize=8.3, leading=13,
                               textColor=colors.HexColor("#8A3A1E")),
    }


def _link_html(url: str, label: str) -> str:
    """reportlab Paragraph 안에서 클릭 가능한 하이퍼링크로 렌더링되는 XML 조각. 정부
    보도자료·기업 공문 톤에 맞춰 색으로 강조하지 않고, 본문과 같은 크기의 진회색 밑줄
    텍스트로 표시한다."""
    if not url:
        return ""
    return f' <link href="{url}" color="#333333"><u>{label}</u></link>'


def _kv_table(rows, widths, st, header=None):
    """라벨-값 표. header가 있으면 옅은 회색 헤더행 추가(정부 보도자료 스타일 - 색 배경 대신
    회색+검정 굵은 글씨로 강조)."""
    data = []
    if header:
        data.append([Paragraph(f"<b>{h}</b>", ParagraphStyle(
            "th", fontName=FONT_B, fontSize=9, leading=13,
            textColor=INK)) for h in header])
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
        style += [("BACKGROUND", (0, 0), (-1, 0), GRAY_HEADER),
                  ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK)]
        start = 1
    style.append(("BACKGROUND", (0, start), (0, -1), GRAY_BG))
    t.setStyle(TableStyle(style))
    return t


def _section_title(num: int, title: str, st):
    """"N. 제목" 형식. 원문자나 박스형 강조 없이 순수 숫자 목록("1." "2." ...)으로
    통일해 정부 보도자료·공문 톤의 절제된 섹션 구분을 유지한다."""
    return Paragraph(f"{num}.　<b>{title}</b>",
                     ParagraphStyle("sectitle", parent=st["sec"], spaceAfter=6))


def _gov_bullets(items, st):
    """정부 보도자료 스타일 개조식 불릿. items는 (level, text) 리스트 -
    level 0='□'(요지), level 1='-'(부연설명, 들여쓰기). 부연설명은 대시(-)로 표시한다 -
    숫자를 쓰면 "1. 2. 3." 목록(extra_reasons 등 별도 번호 목록)과 헷갈릴 수 있어
    여기는 대시를 쓴다."""
    flow = []
    for level, text in items:
        if level == 0:
            style = ParagraphStyle("gov0", parent=st["body"], spaceAfter=2)
            flow.append(Paragraph(f"□　{text}", style))
        else:
            style = ParagraphStyle("gov1", parent=st["small"], leftIndent=5 * mm, spaceAfter=2)
            flow.append(Paragraph(f"-　{text}", style))
    return flow


def _headline_summary(items, st):
    """교육부 등 정부 보도자료 1페이지 상단에 오는 "주요 내용" - 표(라벨:값)가 아니라
    □(요지 문장)/ㅇ(부연) 개조식 서술형 요약. 수신 정보 바로 다음, 문서 맨 위에 위치시켜
    본문을 읽지 않고도 제안 핵심(누구에게·무엇을·얼마에)이 한눈에 들어오게 한다.
    items는 _gov_bullets와 같은 (level, text) 리스트."""
    flow = _gov_bullets(items, st)
    header = Paragraph("주요 내용", ParagraphStyle(
        "sumtitle", fontName=FONT_B, fontSize=9, leading=12, textColor=NAVY))
    wrapper = Table([[header], [flow]], colWidths=[None])
    wrapper.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1.1, NAVY),
        ("TOPPADDING", (0, 0), (-1, 0), 5),
        ("LEFTPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
        ("LEFTPADDING", (0, 1), (-1, 1), 8),
        ("RIGHTPADDING", (0, 1), (-1, 1), 8),
    ]))
    return wrapper


# ⚠️ buyer["왜"]/확인된_사실은 소스가 여러 곳이다 - valuation.py의 정적 BUYERS 목록,
# buyer_lookup.discover_buyers()의 실시간 검색 결과, buyer_lookup.fetch_buyer_disclosure()의
# DeepSeek 요약. 앞의 둘은 소스 자체를 개조식(□/ㅇ)으로 쓰도록 고쳤지만, DeepSeek 요약은
# LLM이 매 요청 새로 생성하는 자연어라 완벽히 통제할 수 없다 - 여기서 최종 방어선으로
# 한 번 더 다듬는다(완결형 종결어미 -> 명사형, 대시 연결 -> 콜론/마침표).
_GAEJOSIK_ENDINGS = [
    (re.compile(r"합니다(?=[.\s]|$)"), "함"),
    (re.compile(r"됩니다(?=[.\s]|$)"), "됨"),
    (re.compile(r"입니다(?=[.\s]|$)"), "임"),
    (re.compile(r"습니다(?=[.\s]|$)"), "음"),
]


def _to_gaejosik(text: str) -> str:
    """자유 문장을 개조식(명사형 종결)에 가깝게 다듬는 최종 보정 - 문장 구조 자체를
    바꾸진 못하지만, 흔한 완결형 종결어미와 대시(—) 연결을 정리한다."""
    if not text:
        return text
    t = re.sub(r"\s*—\s*", ": ", text)
    for pat, repl in _GAEJOSIK_ENDINGS:
        t = pat.sub(repl, t)
    return t


# ---------------- 전문용어 각주 ----------------
# 본문 문장 안에 괄호 설명을 반복하면 가독성이 떨어지므로, 전문용어 뒤에는 별표(*)만
# 달고 문서 맨 끝(유의사항 다음)에 한 번에 몰아서 설명하는 각주 방식을 쓴다. 괄호 안
# 세부조건/부가정보도 같은 이유로 본문 문장에서 빼서 바로 아래 "-" 하위 항목으로
# 옮긴다(예: "잔여수명 800사이클(신품 대비 71%)" -> "잔여수명 800사이클" +
# "- 신품 대비 71%").
GLOSSARY = [
    ("건강도", "SOH(State of Health). 신품 대비 배터리 성능 잔존 비율"),
    ("잔여수명", "RUL(Remaining Useful Life). AI가 예측한 잔여 구동 가능 사이클"),
    ("정전류 구간", "CC(Constant Current). 배터리가 열화될수록 이 구간이 짧아짐"),
    ("AI 진단 파이프라인", "Agent1(화재·안전 위험 게이트) → Agent2(SOH 등급 분류) → "
                       "Agent3(잔여수명·가치 평가) 3단계로 구성"),
    ("RandomForest", "다수의 결정 트리를 결합해 예측 정확도를 높이는 머신러닝 앙상블 모델"),
    ("BMS", "Battery Management System(배터리 관리 시스템)"),
]


def _mark(term: str) -> str:
    """전문용어 뒤에 각주 표시(*)를 붙인다 - GLOSSARY에 같은 용어로 설명을 등록해둬야 한다."""
    return f"{term}*"


def _glossary_block(st):
    """문서 맨 끝(유의사항 다음)에 붙는 전문용어 각주 - 본문에 흩어져 있던 괄호 설명을
    여기 한 곳으로 모았다."""
    flow = [Spacer(1, 4 * mm),
            Paragraph("용어 설명", ParagraphStyle("glosstitle", parent=st["cellb"], spaceAfter=2))]
    for term, desc in GLOSSARY:
        flow.append(Paragraph(f"* {term}: {desc}",
                              ParagraphStyle("gloss", parent=st["small"], spaceAfter=1.5)))
    return flow


def _render_freeform_gov(text: str, st):
    """buyer["왜"] 렌더링 전용. 이미 "□ ...\\nㅇ ..." 개조식으로 줄바꿈되어 있으면(정적
    BUYERS·discover_buyers 소스가 이렇게 씀) 그 구조를 살려 _gov_bullets와 같은 스타일로
    나눠 그린다. 그런 구조가 아닌 자유 문장(DeepSeek 실시간 요약 등)이면 _to_gaejosik로
    한 번 다듬어 한 문단으로 렌더링한다. 반환값은 flowable 리스트."""
    lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
    if lines and any(ln.startswith(("□", "-", "○")) for ln in lines):
        flow = []
        for ln in lines:
            if ln.startswith("□"):
                content = ln[1:].strip("　 ")
                flow.append(Paragraph(f"□　{content}",
                                      ParagraphStyle("gov0f", parent=st["body"], spaceAfter=2)))
            elif ln.startswith(("-", "○")):  # "○"는 구버전 소스 데이터 호환용
                content = ln[1:].strip("　 ")
                flow.append(Paragraph(f"-　{content}",
                                      ParagraphStyle("gov1f", parent=st["small"],
                                                     leftIndent=5 * mm, spaceAfter=2)))
            else:
                flow.append(Paragraph(_to_gaejosik(ln), st["body"]))
        return flow
    return [Paragraph(_to_gaejosik(text or ""), st["body"])]


def _draw_header(canv, PW, PH, ML, MR, HEADER_H, doc_no, today, seller, title_text, subtitle_en):
    """모든 페이지 상단에 반복되는 헤더 - MijungE 로고 lockup + 헤드라인 + 문서정보 표."""
    canv.saveState()

    # 로고 lockup: "MijungE" + 구분선 + 서브텍스트
    canv.setFillColor(NAVY)
    canv.setFont(FONT_B, 15)
    canv.drawString(ML, PH - 11 * mm, "MijungE")
    logo_w = canv.stringWidth("MijungE", FONT_B, 15)
    canv.setStrokeColor(GRAY_LINE)
    canv.setLineWidth(0.7)
    canv.line(ML + logo_w + 3 * mm, PH - 13.3 * mm, ML + logo_w + 3 * mm, PH - 8.7 * mm)
    canv.setFillColor(GRAY_TXT)
    canv.setFont(FONT_R, 8.3)
    canv.drawString(ML + logo_w + 6 * mm, PH - 11 * mm, "배터리진단팀")

    # 헤드라인 타이틀 + 영문 서브타이틀
    canv.setFillColor(INK)
    canv.setFont(FONT_B, 17)
    canv.drawString(ML, PH - 22 * mm, title_text)
    canv.setFillColor(GRAY_TXT)
    canv.setFont(FONT_R, 8.3)
    canv.drawString(ML, PH - 27 * mm, subtitle_en)

    # 문서정보 표(우측) - 표 형식(라벨 셀 회색/값 셀 흰색). 담당자 실명·이메일을 별도 행으로
    # 추가 - 문의 창구가 있는 실제 회사 문서처럼 보이게 하기 위함.
    info_tbl = Table([
        ["문서번호", doc_no],
        ["작성일자", f"{today:%Y-%m-%d}"],
        ["작 성 자", seller],
        ["담당자", f"{CONTACT_NAME} ({CONTACT_EMAIL})"],
    ], colWidths=[20 * mm, 55 * mm], rowHeights=[6.3 * mm] * 4)
    info_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), GRAY_HEADER),
        ("BACKGROUND", (1, 0), (1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, GRAY_LINE),
        ("FONT", (0, 0), (0, -1), FONT_B, 7.6),
        ("FONT", (1, 0), (1, -1), FONT_R, 7.6),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TEXTCOLOR", (0, 0), (0, -1), INK),
        ("TEXTCOLOR", (1, 0), (1, -1), INK),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    tw, th = info_tbl.wrapOn(canv, 0, 0)
    info_tbl.drawOn(canv, PW - MR - tw, PH - 9 * mm - th)

    # 헤더-바디 구분선
    canv.setStrokeColor(NAVY)
    canv.setLineWidth(1.1)
    canv.line(ML, PH - HEADER_H, PW - MR, PH - HEADER_H)
    canv.restoreState()


def _draw_footer(canv, PW, MR, ML, footer_text):
    canv.saveState()
    canv.setStrokeColor(GRAY_LINE)
    canv.setLineWidth(0.5)
    canv.line(ML, 14 * mm, PW - MR, 14 * mm)
    canv.setFont(FONT_R, 7.6)
    canv.setFillColor(GRAY_TXT)
    canv.drawString(ML, 9.5 * mm, footer_text)
    canv.drawRightString(PW - MR, 9.5 * mm, f"- {canv.getPageNumber()} -")
    canv.restoreState()


def build_pdf(*, buyer: dict, capacity_kwh: float, grade: str,
              rul_cycles: float, health_pct: float, indicators: dict,
              full_life: float, won, seller: str = "배터리 진단 AI 시스템",
              doc_no: str | None = None, eco: dict | None = None,
              fire_note: str = "", extra_reasons: list[str] | None = None) -> bytes:
    """매도 제안서 PDF 바이트 반환."""
    _register_fonts()
    st = _styles()
    today = date.today()
    doc_no = doc_no or f"BAT-{today:%Y%m%d}-{abs(hash(buyer['매입처'])) % 10000:04d}"

    buf = io.BytesIO()
    PW, PH = A4
    ML, MR = 18 * mm, 18 * mm
    HEADER_H = 38 * mm

    def page(canv, _doc):
        _draw_header(canv, PW, PH, ML, MR, HEADER_H, doc_no, today, seller,
                     "사용후 배터리 매도 제안서", "Used EV Battery Sales Proposal")
        _draw_footer(canv, PW, MR, ML, "사용후 배터리 매도 제안서 · 대외비(Confidential) · 문의: tkyaho@mijungev.kro.kr")

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
    S.append(Spacer(1, 6 * mm))

    # ── 주요 내용 요약 (교육부 등 정부 보도자료 1페이지 스타일 - □(대분류)마다 ㅇ 여러 줄인
    # 실제 보도자료 구조를 따라 매도 개요/AI 진단/제안 가격/매입처 적합성 4개 블록으로 확장)
    lo, hi = buyer["제안가_범위_원"]
    _why_gist = (buyer["왜"].split("\n")[0].lstrip("□").strip("　 ")
                if buyer.get("왜") else "")
    S.append(_headline_summary([
        (0, f"{buyer['매입처']}에 사용후 배터리 {capacity_kwh:g}kWh 매도 제안"),
        (1, f"매입처: {buyer['매입처']}"),
        (1, f"업종: {buyer['역할']}"),
        (1, f"단가대: {buyer['단가대']}"),
        (0, "AI 진단 결과"),
        (1, f"판별등급 {grade}, AI {_mark('진단 건강도')} {health_pct:.1f}%"),
        (1, f"{_mark('예측 잔여수명')} {rul_cycles:,.0f}사이클"),
        (1, f"신품 대비 {rul_cycles / full_life * 100:.0f}%({full_life:,.0f}사이클 기준)"),
        (0, "제안 가격"),
        (1, f"제안총액 {won(buyer['제안가_원'])}, 협의범위 {won(lo)}~{won(hi)}"),
        (1, f"적용단가 {buyer['단가_원per_kWh']:,}원/kWh"),
    ] + ([(0, "매입처 적합성"), (1, _why_gist)] if _why_gist else []), st))
    S.append(Spacer(1, 8 * mm))

    # ── ① 제안 가격
    S.append(_section_title(1, "제안 가격", st))
    S.append(Spacer(1, 3 * mm))
    S.append(Paragraph("제안 총액", ParagraphStyle(
        "pl", fontName=FONT_R, fontSize=9, textColor=GRAY_TXT)))
    S.append(Paragraph(f"<b>{won(buyer['제안가_원'])}</b>", ParagraphStyle(
        "pv", fontName=FONT_B, fontSize=24, leading=29, textColor=NAVY)))
    S.append(Spacer(1, 3 * mm))
    S.append(_kv_table([
        ["적용 단가", f"{buyer['단가_원per_kWh']:,} 원 / kWh　({buyer['단가대']})"],
        ["협의 범위", f"{won(lo)}　~　{won(hi)}"],
        ["공칭 용량", f"{capacity_kwh:g} kWh"],
    ], [28 * mm, CW - 28 * mm], st))
    S.append(Spacer(1, 3 * mm))
    price_link = _link_html(buyer.get("단가출처_링크", ""), buyer.get("단가출처_라벨") or "참고자료")
    S.append(Paragraph(f"※ 단가 근거: {buyer['단가근거']}{price_link}", st["small"]))
    if buyer.get("등급제한_적용"):
        S.append(Spacer(1, 1.5 * mm))
        S.append(Paragraph(
            f"※ 귀사 최고 취급 단가대: {buyer['매입처_최고단가대']}. 본 배터리 진단 등급 "
            f"'{grade}' 기준 {buyer['단가대']} 단가 적용 제안", st["note"]))
    S.append(Spacer(1, 8 * mm))

    # ── ② 배터리 상태 진단
    S.append(_section_title(2, "배터리 상태 진단 (AI 진단 결과)", st))
    S.append(Spacer(1, 3 * mm))
    S.append(_kv_table([
        ["판별 등급", f"<b>{grade}</b>"],
        ["예측 잔여수명", f"<b>{rul_cycles:,.0f} 사이클</b>　(신품 기준 {full_life:,.0f} 사이클)"],
        ["추정 건강도", f"<b>{health_pct:.1f} %</b>"],
    ], [32 * mm, CW - 32 * mm], st))
    S.append(Spacer(1, 4 * mm))

    IND_KO = {"life": "수명 여유", "capacity": "방전 지속력",
              "charge": "충전 건전성", "stability": "전압 안정성"}
    ind_rows = [[IND_KO.get(k, k), f"{v*100:.0f} / 100"] for k, v in indicators.items()]
    # KeepTogether - 표가 페이지 경계에서 헤더행과 본문행이 분리되는 걸 방지한다.
    S.append(KeepTogether(_kv_table(ind_rows, [CW * 0.5, CW * 0.5], st,
                                    header=["건전성 세부 지표", "점수"])))
    S.append(Spacer(1, 3 * mm))
    for f in _gov_bullets([
        (0, f"진단 방식: 충·방전 센서값을 {_mark('RandomForest')} 회귀·분류 모델로 분석"),
        (1, "잔여수명 예측 평균오차 ±11 사이클"),
        (1, "등급 판별 정확도 98.4%"),
    ], st):
        S.append(f)
    S.append(Spacer(1, 4 * mm))

    # 등급 판정 기준 — SOH(추정 건강도) + 전압 안정성 지표를 함께 보고
    # 매도 경로 상한(재사용/2차사용/재활용)을 정하는 기준을 명시한다.
    # 정부 보도자료 스타일 □/ㅇ 개조식으로 항목을 잘게 쪼갠다(2026-08 개편).
    # KeepTogether로 묶어 표가 페이지 경계에서 잘리지 않게 한다.
    S.append(KeepTogether([
        Paragraph("등급 판정 기준", st["cellb"]),
        Spacer(1, 1.5 * mm),
        _kv_table([
            ["SOH 80% 이상 + 전압 안정성 우수", "재사용(EV 재제조)급"],
            ["SOH 60~80% + 안정성 확보", "2차사용(ESS)급"],
            ["SOH 60% 미만 또는 이상 징후", "재활용(소재회수)급"],
        ], [CW * 0.62, CW * 0.38], st, header=["평가 결과", "판정"]),
        Spacer(1, 3 * mm),
    ] + _gov_bullets([
        (0, "SOH 대역 매핑"),
        (1, "SOH는 데이터에 없어 등급별 SOH 대역(1등급 80~100%, 2등급 60~80%)에 진단 건강도 매핑"),
        (1, "3등급(재활용)은 하한 없음: 배터리 분해·용해로 원재료 추출하는 공정, SOH 성능과 무관하게 처리 가능"),
        (1, "실제 문헌에도 재활용에 성능 하한을 두는 사례 없음"),
        (1, "전압 안정성은 건전성 세부 지표의 '전압 안정성' 점수 기준 판단"),
    ], st) + [Spacer(1, 2 * mm)] + _gov_bullets([
        (0, "판정 기준 출처"),
        (1, "SOH 60%·80% 컷오프: 이차전지 재사용 업계 자료(ROPLANT) 및 한국에너지경제연구원(KESRC) 정책연구"),
        (1, "ROPLANT 기준: 80~90% 이상 재제조, 60~80% 재사용, 60% 미만 재활용"),
        (1, "KESRC 기준: 완성차 성능보증 통상 70~80%, ESS 재사용 시 초기용량 60%까지 사용 후 폐기 가능"),
        (1, "※ 문헌별 명칭 상이 사례: 한국소방안전원(KFPA)은 80% 이상 '재사용', 65~80% '재제조'로 표기"),
        (1, "본 제안서는 고SOH 구간을 재제조(EV 재장착)급으로 정의(근거: 차량 재장착은 타 용도 전용보다 높은 건강 상태 요구)"),
        (1, "정부는 탈거 전 성능평가 거쳐 재제조·재사용 가능 배터리를 순환자원으로 지정하는 제도 추진 중"),
        (1, "본 등급 체계는 위 정책 방향과 연동 가능하도록 설계"),
    ], st) + [Spacer(1, 2 * mm)] + _gov_bullets([
        (0, "재활용 등급 내 이상징후 취급"),
        (1, "재활용은 SOH 하한 없음. 단, 물리적 손상·열폭주 전조 등 안전 이상은 별도 감지"),
        (1, f"{_mark('화재 위험 게이트')} 위험 감지 시 SOH 등급 무관 즉시 폐기·특별취급 대상 분류"),
        (1, "위험 감지 시 후속 등급 판정은 진행하지 않음"),
        (1, "실무 문헌은 셀 단위 계측 권장"),
        (1, "측정 항목 예시: 내부저항 증가율, 셀간 전압편차"),
        (1, "본 진단은 비침습 방식: 팩 단위 충·방전 센서값 기반 전압 강하 패턴 지표 사용"),
        (1, f"셀 단위 IR·전압편차 계측은 팩 분해 또는 {_mark('BMS')} 직접 연동 필요"),
        (1, "본 진단 범위(비분해 검사) 밖의 정밀 진단 항목으로 별도 분류"),
    ], st)))
    if fire_note:
        S.append(Spacer(1, 2 * mm))
        S.append(Paragraph(f"안전성: {fire_note}", st["body"]))
    S.append(Spacer(1, 8 * mm))

    # ── ③ 경제성·환경 효과 (선택)
    if eco:
        S.append(_section_title(3, "경제성 · 환경 효과", st))
        S.append(Spacer(1, 3 * mm))
        # 신품 대체가 성립하는 경로(재사용·2차사용)에서만 절감액을 제시한다
        if eco.get("절감_적용", True):
            _cost_rows = [
                ["구매자 절감액", f"<b>{won(eco['구매자절감_원'])}</b>　"
                              f"(신품 대비 {eco['절감률_퍼센트']:.1f}% 절감)"],
                ["신품 등가 비용", won(eco["신품등가비용_원"])],
            ]
        else:
            _cost_rows = [["신품 대비 절감", "해당 없음: " + eco.get("절감_미적용사유", "")]]
        # 3등급(재활용)은 '실사용 kWh'도 SOH 하한 없는 값에서 파생된 수치라
        # 같이 감춘다 — 애초에 재활용 CO2 계산에도 이 값을 쓰지 않는다(용량 무관 20% 고정).
        _soh_cell = ("60% 미만　(재활용은 용량 무관 처리: 실사용 kWh 산정 불필요)"
                    if eco.get("SOH_하한_근거없음")
                    else f"{eco['추정_SOH_퍼센트']} %　(실사용 {eco['실사용_kWh']} kWh)")
        S.append(_kv_table(_cost_rows + [
            ["추정 SOH", _soh_cell],
            # ⚠️ 유니코드 첨자 "₂"(U+2082)는 번들 폰트(NotoSansKR)에 글리프가 없어
            # PDF에서 빈 네모(tofu)로 깨졌다 — 코드베이스 어디서든 CO2e 표기는 일반
            # 숫자 "2"만 쓴다(economics.py의 모든 "CO2e" 표기와 통일).
            ["CO2 절감량", f"<b>{eco['탄소절감_kgCO2e']:,.0f} kgCO2e</b>　"
                        f"(승용차 {eco['승용차_년_환산']:.2f}년 배출 / 소나무 "
                        f"{eco['소나무_그루_환산']:,}그루 연간 흡수량)"],
        ], [32 * mm, CW - 32 * mm], st))
        S.append(Spacer(1, 3 * mm))
        S.append(Paragraph(
            "탄소 계수: 제조 탄소발자국 Nature Communications(2024), "
            "재제조 공정 배출 3% iScience(2023) 적용", st["small"]))
        S.append(Spacer(1, 8 * mm))
        n4, n5 = 4, 5
    else:
        n4, n5 = 3, 4

    # ── 귀사 적합성
    fit_block = [
        _section_title(n4, "귀사 적합성", st),
        Spacer(1, 3 * mm),
    ] + _render_freeform_gov(buyer["왜"], st)
    # extra_reasons — 화면("배터리 매도 제안서" 탭)에 이미 표시된 추가 사유를 그대로
    # 덧붙인다. 화면·PDF 내용이 어긋나지 않게 하기 위함. buyer["왜"]의 □/○ 불릿과 헷갈리지
    # 않도록 번호("1. 2. 3.") 목록으로 구분하고, st["small"]로 폰트 크기를 나머지 개조식
    # 텍스트와 맞춘다(예전엔 st["body"]라 더 크게 나와 폰트가 안 맞아 보였다). _to_gaejosik로
    # 완결형 종결어미·대시도 한 번 더 정리한다.
    for i, r in enumerate(extra_reasons or [], start=1):
        fit_block.append(Spacer(1, 1.5 * mm))
        fit_block.append(Paragraph(f"{i}. {_to_gaejosik(r)}",
                                   ParagraphStyle("numfit", parent=st["small"], leftIndent=5 * mm)))
    buyer_link = _link_html(buyer.get("출처_링크", ""), "참고자료")
    fit_block += [
        Spacer(1, 2 * mm),
        Paragraph(f"확인된 사업 영역: {_to_gaejosik(buyer['확인된_사실'])}{buyer_link}", st["small"]),
    ]
    S.append(KeepTogether(fit_block))
    S.append(Spacer(1, 8 * mm))

    # ── 유의사항
    # ⚠️ 이전엔 "『사용후배터리 산업 육성법』(2025.10 시행)"이라고 적혀 있었는데 사실과 달랐다 -
    # 이 법(사용후 배터리의 관리 및 산업육성에 관한 법률)은 2026.5.26 공포, 2027.5.27 시행
    # "예정"이라 아직 시행 전이다(웹서치로 뉴스 보도 확인). 실제로 지금 적용되는 근거는
    # 「자원순환기본법」상 순환자원 지정 고시(전기차 폐배터리는 폐기물관리법 규제 면제 대상이지만
    # 준수사항이 있음)와, 수출 시 「폐기물의 국가 간 이동 및 그 처리에 관한 법률」(바젤협약 국내
    # 이행법)이다 - 실제 회사에 제출하는 문서라 법령 인용은 정확해야 해서 전부 다시 검증했다.
    # (아래 두 번째 항목의 「사용후 배터리의 관리 및 산업육성에 관한 법률」은 실제 법률 제목이라
    # 용어 통일 대상에서 제외 - "사용후"를 "재사용"으로 바꾸면 법령명을 오인용하게 된다.)
    S.append(_section_title(n5, "유의사항", st))
    S.append(Spacer(1, 3 * mm))

    def _gov_link(text, url, label):
        link = _link_html(url, label) if url else ""
        return Paragraph(f"-　{text}{link}",
                         ParagraphStyle("gov1link", parent=st["small"], leftIndent=5 * mm, spaceAfter=2))

    S.append(Paragraph("□　본 제안가 산정 근거", ParagraphStyle("gov0", parent=st["body"], spaceAfter=2)))
    S.append(_gov_link("공개 실거래·시장 벤치마크 + AI 진단 결과 결합 추정치", "", ""))
    S.append(_gov_link("귀사가 제시한 견적 아님", "", ""))
    S.append(Spacer(1, 2 * mm))

    S.append(Paragraph("□　가격 조정 가능성", ParagraphStyle("gov0", parent=st["body"], spaceAfter=2)))
    S.append(_gov_link("최종 가격은 실물 검사(외관·전기적 검사) 및 시황에 따라 조정 가능", "", ""))
    S.append(Spacer(1, 2 * mm))

    S.append(Paragraph("□　「자원순환기본법」 순환자원 지정 고시 대상", ParagraphStyle("gov0", parent=st["body"], spaceAfter=2)))
    S.append(_gov_link("전기차 사용후 배터리: 폐기물관리법 규제 면제",
                       "https://www.korea.kr/news/policyNewsView.do?newsId=148905610", "정책브리핑(정부) 자료"))
    S.append(_gov_link("단, 단순 수리·수선·건조·세척 등 일반적·품목별 준수사항 충족 필요", "", ""))
    S.append(_gov_link("미충족 시 폐기물처리업 허가 대상 가능", "", ""))
    S.append(Spacer(1, 2 * mm))

    S.append(Paragraph("□　「사용후 배터리의 관리 및 산업육성에 관한 법률」 제정", ParagraphStyle("gov0", parent=st["body"], spaceAfter=2)))
    S.append(_gov_link("2026.5.26 공포, 2027.5.27 시행 예정(아직 시행 전)",
                       "https://zdnet.co.kr/view/?no=20260520090026", "관련 보도"))
    S.append(_gov_link("시행 후 탈거 전 성능평가·등급분류 및 전주기 이력·거래시스템 등록 의무화 예정", "", ""))
    S.append(_gov_link("사전 대비 필요", "", ""))
    S.append(Spacer(1, 2 * mm))

    S.append(Paragraph("□　해외 매입처 매도(수출) 시 유의", ParagraphStyle("gov0", parent=st["body"], spaceAfter=2)))
    S.append(_gov_link("「폐기물의 국가 간 이동 및 그 처리에 관한 법률」(바젤협약 국내 이행법) 적용", "", ""))
    S.append(_gov_link("사전통보·승인 절차 별도 이행 필요", "", ""))

    # ── 용어 설명 (전문용어 각주 - 본문의 "*" 표시된 용어를 여기서 한 번에 설명)
    for f in _glossary_block(st):
        S.append(f)

    # ── 서명란
    S.append(Spacer(1, 10 * mm))
    sign = Table([["제 안 자", CONTACT_NAME, "( 인 )"]],
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


def build_pdf_from_view(*, buyer_name: str = "매입 희망 기업", buyer_role: str = "",
                        buyer_location: str = "",
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
    HEADER_H = 38 * mm

    def page(canv, _doc):
        _draw_header(canv, PW, PH, ML, MR, HEADER_H, doc_no, today, seller,
                     "사용후 배터리 매도 제안서", "Used EV Battery Sales Proposal")
        _draw_footer(canv, PW, MR, ML, "사용후 배터리 매도 제안서 · 대외비(Confidential) · 문의: tkyaho@mijungev.kro.kr")

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
    S.append(Spacer(1, 6 * mm))

    # ── 주요 내용 요약 (교육부 등 정부 보도자료 1페이지 스타일 - □(대분류)마다 ㅇ 여러 줄인
    # 실제 보도자료 구조를 따라 매도 개요/AI 진단/제안 가격/귀사 적합성 4개 블록으로 확장)
    _reasons_gist = (reasons[0].split("\n")[0].lstrip("□○-").strip("　 ")
                    if reasons else "")
    S.append(_headline_summary([
        (0, f"{buyer_name}에 사용후 배터리 매도 제안"),
        (1, f"매입처: {buyer_name}"),
    ] + ([(1, f"업종: {buyer_role}")] if buyer_role else []) + [
        (0, "AI 진단 결과"),
        (1, f"판별등급 {grade}, AI {_mark('진단 건강도')} {health_score_pct:.1f}%"),
        (1, f"{_mark('예측 잔여수명')} {remaining_cycle:,.0f}사이클"),
        (1, f"신품 대비 {remaining_cycle / new_cycle * 100:.0f}%({new_cycle:,.0f}사이클 기준)"),
        (0, "제안 가격"),
        (1, f"제안총액 {price_total_manwon:,.0f}만원, 협의범위 {negotiation_range}"),
        (1, f"적용단가 {unit_price_won:,.0f}원/kWh({price_grade_label})"),
    ] + ([(0, "귀사 적합성"), (1, _reasons_gist)] if _reasons_gist else []), st))
    S.append(Spacer(1, 8 * mm))

    S.append(_section_title(1, "제안 가격", st))
    S.append(Spacer(1, 3 * mm))
    S.append(Paragraph("제안 총액", ParagraphStyle(
        "pl", fontName=FONT_R, fontSize=9, textColor=GRAY_TXT)))
    S.append(Paragraph(f"<b>{price_total_manwon:,.0f}만원</b>", ParagraphStyle(
        "pv", fontName=FONT_B, fontSize=24, leading=29, textColor=NAVY)))
    S.append(Spacer(1, 3 * mm))
    S.append(_kv_table([
        ["적용 단가", f"{unit_price_won:,.0f} 원 / kWh　({price_grade_label})"],
        ["협의 범위", negotiation_range],
    ], [28 * mm, CW - 28 * mm], st))
    if price_note:
        S.append(Spacer(1, 3 * mm))
        S.append(Paragraph(f"※ {price_note}", st["small"]))
    S.append(Spacer(1, 8 * mm))

    S.append(_section_title(2, "배터리 상태 진단 (AI 진단 결과)", st))
    S.append(Spacer(1, 3 * mm))
    S.append(_kv_table([
        ["판별 등급", f"<b>{grade}</b>"],
        ["예측 잔여수명", f"<b>{remaining_cycle:,.0f}</b> 사이클　(신품 기준 {new_cycle:,.0f} 사이클)"],
        ["추정 건강도", f"<b>{health_score_pct:.1f}%</b>"],
    ], [32 * mm, CW - 32 * mm], st))
    S.append(Spacer(1, 4 * mm))

    if health_metrics:
        S.append(KeepTogether(_kv_table(
            [[m.get("label", ""), m.get("score", "")] for m in health_metrics],
            [CW * 0.5, CW * 0.5], st, header=["건전성 세부 지표", "점수"])))
        S.append(Spacer(1, 3 * mm))
    if diagnosis_note:
        S.append(Paragraph(diagnosis_note, st["small"]))
    S.append(Spacer(1, 8 * mm))

    # reasons/cautions는 build_pdf()의 extra_reasons와 동일하게 번호("1. 2. 3.") 목록 +
    # st["small"] 폰트 + _to_gaejosik 보정을 적용한다(두 함수 렌더링 스타일 통일).
    if reasons:
        S.append(_section_title(3, "귀사에 적합한 이유", st))
        S.append(Spacer(1, 3 * mm))
        for i, r in enumerate(reasons, start=1):
            S.append(Paragraph(f"{i}. {_to_gaejosik(r)}",
                               ParagraphStyle("numfv", parent=st["small"], leftIndent=5 * mm)))
            S.append(Spacer(1, 1.5 * mm))
        S.append(Spacer(1, 6.5 * mm))

    if cautions:
        S.append(_section_title(4, "유의사항", st))
        S.append(Spacer(1, 3 * mm))
        for i, c in enumerate(cautions, start=1):
            S.append(Paragraph(f"{i}. {_to_gaejosik(c)}",
                               ParagraphStyle("numcv", parent=st["small"], leftIndent=5 * mm)))
            S.append(Spacer(1, 1.5 * mm))

    # ── 용어 설명 (전문용어 각주 - 본문의 "*" 표시된 용어를 여기서 한 번에 설명)
    for f in _glossary_block(st):
        S.append(f)

    S.append(Spacer(1, 10 * mm))
    sign = Table([["제 안 자", CONTACT_NAME, "( 인 )"]],
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
