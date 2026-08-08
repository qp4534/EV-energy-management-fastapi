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
                    f"아래는 '{buyer_name}'을 검색한 결과다. 이 회사의 폐배터리/이차전지 "
                    f"재사용·재활용 관련 최신 공개 사업 현황을 한국어로 2문장 이내로 요약해줘. "
                    f"검색 결과에 관련 내용이 없으면 일반적인 사업 영역만 간단히 설명해.\n\n{context}"
                ),
            }],
            "max_tokens": 300,
            "temperature": 0.3,
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
    """buyer_name 기업의 폐배터리 관련 최신 공개 사업 현황을 검색+요약해 한국어 2문장
    이내로 돌려준다. 실패하면(키 없음/검색 실패/요약 실패) None."""
    s_key = serper_api_key or os.environ.get("SERPER_API_KEY")
    d_key = deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not s_key or not d_key:
        return None
    try:
        snippets = _search_serper(f"{buyer_name} 폐배터리 이차전지 재사용", s_key)
        return _summarize_deepseek(buyer_name, snippets, d_key)
    except Exception:
        # 검색/네트워크/키 문제 등 무엇이든 실패하면 정적 문구 폴백으로 넘어간다.
        return None
