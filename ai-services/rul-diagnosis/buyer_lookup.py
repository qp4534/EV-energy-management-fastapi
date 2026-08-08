# -*- coding: utf-8 -*-
"""매입처 실시간 공시/뉴스 조회 (선택 기능).

valuation.py의 BUYERS 리스트에 있는 '왜'/'확인된 사실' 문구는 만들 때 미리 조사해서
넣어둔 고정 텍스트다(요청마다 검색하는 게 아니었음). 이 모듈은 요청 시점에 Claude의
web_search 도구로 매입처의 최신 공개 정보를 찾아 그 문구를 보강한다.

ANTHROPIC_API_KEY가 없거나 검색/호출이 실패하면 None을 반환한다 - 호출부는 반드시
valuation.py의 정적 문구로 폴백해야 한다(제안서 생성 자체가 이 기능 때문에 막히면 안 됨).
"""
import os


def fetch_buyer_disclosure(buyer_name: str, api_key: str | None = None) -> str | None:
    """buyer_name 기업의 폐배터리 관련 최신 공개 사업 현황을 웹 검색으로 찾아
    한국어 2문장 이내로 요약해 돌려준다. 실패하면 None."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=300,
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}],
            messages=[{
                "role": "user",
                "content": (
                    f"'{buyer_name}'의 폐배터리/이차전지 재사용·재활용 관련 최신 공개 "
                    f"사업 현황을 웹에서 찾아 한국어로 2문장 이내로 요약해줘. 출처가 "
                    f"불분명하면 일반적인 사업 영역만 간단히 설명해."
                ),
            }],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or None
    except Exception:
        # 검색/네트워크/키 문제 등 무엇이든 실패하면 정적 문구 폴백으로 넘어간다.
        return None
