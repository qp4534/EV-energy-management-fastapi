# -*- coding: utf-8 -*-
"""매입처 실시간 공시/뉴스 조회 (선택 기능).

valuation.py의 BUYERS 리스트에 있는 '왜'/'확인된 사실' 문구는 만들 때 미리 조사해서
넣어둔 고정 텍스트다(요청마다 검색하는 게 아니었음). 이 모듈은 요청 시점에 매입처의
최신 공개 정보를 찾아 그 문구를 보강한다.

비용을 최소화하려고 Claude 대신 이 조합을 쓴다:
  1. Serper(구글 검색 API)로 실제 웹 검색 — LLM이 아니라 검색 결과만 가져오므로 저렴
  2. DeepSeek(deepseek-chat)로 검색 결과를 한국어 2문장으로 요약 — 토큰 단가가 낮음

SERPER_API_KEY/DEEPSEEK_API_KEY가 없거나(둘 다 있어야 동작) 검색/요약 중 무엇이든
실패하면 None을 반환한다 - 호출부는 반드시 valuation.py의 정적 문구로 폴백해야 한다
(제안서 생성 자체가 이 기능 때문에 막히면 안 됨).
"""
import os

import requests

SERPER_URL = "https://google.serper.dev/search"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def _search_serper(query: str, api_key: str, num: int = 5) -> list[dict]:
    resp = requests.post(
        SERPER_URL,
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query, "gl": "kr", "hl": "ko", "num": num},
        timeout=8,
    )
    resp.raise_for_status()
    return resp.json().get("organic", [])


def _summarize_deepseek(buyer_name: str, snippets: list[dict], api_key: str) -> str | None:
    if not snippets:
        return None
    context = "\n".join(
        f"- {s.get('title', '')}: {s.get('snippet', '')}" for s in snippets[:5]
    )
    resp = requests.post(
        DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "deepseek-chat",
            "messages": [{
                "role": "user",
                "content": (
                    f"아래는 '{buyer_name}'을 검색한 결과다. 이 회사가 사용후 배터리(폐배터리/"
                    f"이차전지 재사용·재활용)를 실제로 매입·수거하겠다고 공개적으로 밝힌 근거"
                    f"(보도자료, 공시, 뉴스, 사업 공고 등)가 있는지 확인해서 한국어 2문장 이내로 "
                    f"요약해줘. 반드시 검색 결과에 실제로 있는 내용만 근거로 삼고, 매입 의사를 "
                    f"밝힌 근거가 검색 결과에 없으면 지어내지 말고 '공개된 자료에서 매입 의사를 "
                    f"직접 확인하지는 못했다'고 솔직히 말해줘.\n\n{context}"
                ),
            }],
            "max_tokens": 300,
            "temperature": 0.2,
        },
        timeout=15,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    return text or None


def fetch_buyer_disclosure(
    buyer_name: str,
    serper_api_key: str | None = None,
    deepseek_api_key: str | None = None,
) -> str | None:
    """buyer_name 기업이 사용후 배터리를 매입하겠다고 밝힌 공개 근거자료(보도자료/공시/뉴스)를
    검색해 한국어 2문장 이내로 요약해 돌려준다. 실패하면(키 없음/검색 실패/요약 실패) None."""
    s_key = serper_api_key or os.environ.get("SERPER_API_KEY")
    d_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not s_key or not d_key:
        return None
    try:
        # "재사용" 같은 일반 사업 소개보다 "매입/수거"처럼 실제 구매 의사를 밝힌 자료를
        # 우선 찾도록 검색어 자체를 구매 의도 중심으로 잡는다.
        snippets = _search_serper(f"{buyer_name} 사용후 배터리 매입 수거 공고", s_key)
        return _summarize_deepseek(buyer_name, snippets, d_key)
    except Exception:
        # 검색/네트워크/키 문제 등 무엇이든 실패하면 정적 문구 폴백으로 넘어간다.
        return None


# 검색으로 찾은 회사를 valuation.py의 BUYERS 항목과 같은 모양으로 만들 때 쓰는 매핑.
# band(단가대)별로 그 단가대에서 통상 받아주는 등급 범위 - 회사가 실제로 어떤 등급까지
# 받는지는 검색으로 알 수 없어서, PRICE_BANDS 위계(수거<재활용<2차사용<재사용)에 맞춰
# 합리적인 기본값을 쓴다(기존 BUYERS 정적 항목들의 accepts 패턴과 동일).
_ACCEPTS_BY_BAND = {
    "reuse": ["1등급", "2등급"],
    "ess": ["1등급", "2등급"],
    "material": ["1등급", "2등급", "3등급"],
    "collect": ["1등급", "2등급", "3등급"],
}
_VALID_BANDS = set(_ACCEPTS_BY_BAND)


def discover_buyers(
    serper_api_key: str | None = None,
    deepseek_api_key: str | None = None,
    num_results: int = 10,
) -> list[dict] | None:
    """국내에서 사용후 배터리를 실제로 매입·수거·재사용·재활용하는 회사를 검색해서
    valuation.BUYERS와 같은 모양(name/emoji/loc/role/band/accepts/why/fact)의 목록으로
    돌려준다. 가격 자체는 여기서 만들지 않는다 - estimate_offers()가 이 목록을 받아서
    기존 PRICE_BANDS(BNEF/국내 낙찰가 등 출처가 있는 벤치마크) 계산식을 그대로 적용한다.

    검색/키/JSON 파싱 중 무엇이든 실패하면 None - 호출부는 반드시 valuation.py의
    고정 BUYERS 목록으로 폴백해야 한다."""
    s_key = serper_api_key or os.environ.get("SERPER_API_KEY")
    d_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not s_key or not d_key:
        return None
    try:
        snippets = _search_serper(
            "국내 사용후 배터리 매입 수거 재사용 재활용 기업", s_key, num=num_results
        )
        if not snippets:
            return None
        context = "\n".join(
            f"- {s.get('title', '')}: {s.get('snippet', '')} ({s.get('link', '')})"
            for s in snippets
        )
        resp = requests.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {d_key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "response_format": {"type": "json_object"},
                "messages": [{
                    "role": "user",
                    "content": (
                        "아래는 '국내 사용후 배터리 매입/수거/재사용/재활용 기업'을 검색한 결과다. "
                        "검색 결과에 실제로 등장하는 회사만 뽑아서(지어내지 말 것) JSON으로 정리해줘. "
                        '형식: {"companies": [{"name": "회사명", "role": "사업 내용 한 줄", '
                        '"loc": "지역 (모르면 \\"—\\")", "band": "reuse|ess|material|collect 중 하나 '
                        "(reuse=EV 재제조/재사용, ess=2차사용·ESS, material=소재회수·재활용, "
                        'collect=단순 수거·매입 중개)", "fact": "검색 결과에 실제로 있는 근거 한 줄"}]}. '
                        "검색 결과에서 확인되지 않는 내용은 절대 지어내지 말고, 근거가 빈약한 회사는 "
                        f"목록에서 빼줘. 최대 8곳까지만.\n\n{context}"
                    ),
                }],
                "max_tokens": 1500,
                "temperature": 0.1,
            },
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        import json
        parsed = json.loads(raw)
        companies = parsed.get("companies") if isinstance(parsed, dict) else None
        if not companies:
            return None

        buyers = []
        for c in companies:
            name = (c.get("name") or "").strip()
            band = (c.get("band") or "").strip()
            if not name or band not in _VALID_BANDS:
                continue
            buyers.append({
                "name": name,
                "emoji": "🔎",
                "loc": c.get("loc") or "—",
                "role": c.get("role") or "",
                "band": band,
                "accepts": _ACCEPTS_BY_BAND[band],
                "why": c.get("role") or "실시간 검색으로 확인된 매입처입니다.",
                "fact": c.get("fact") or "실시간 검색 결과 기반 - 상세 근거는 직접 확인 필요.",
            })
        return buyers or None
    except Exception:
        # 검색/키/JSON 파싱 등 무엇이든 실패하면 고정 BUYERS 목록으로 폴백.
        return None
