import json

import httpx
import pytest

from app.ai.config import AISettings
from app.ai.deepseek import DeepSeekClient, DeepSeekUnavailable


@pytest.mark.asyncio
async def test_deepseek_chat_uses_flash_and_disables_thinking() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "근거 기반 답변"},
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = AISettings(database_url="postgresql+asyncpg://test", deepseek_api_key="secret")
    client = DeepSeekClient(settings, http_client)

    answer = await client.generate("system", "user", purpose="chat")

    assert answer == "근거 기반 답변"
    assert captured["authorization"] == "Bearer secret"
    assert captured["body"]["model"] == "deepseek-v4-flash"
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert "response_format" not in captured["body"]
    await http_client.aclose()


@pytest.mark.asyncio
async def test_deepseek_report_requests_json_without_exposing_reasoning() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "reasoning_content": "must not be returned",
                            "content": '{"summary":"요약"}',
                        },
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = AISettings(database_url="postgresql+asyncpg://test", deepseek_api_key="secret")
    client = DeepSeekClient(settings, http_client)

    result = await client.generate(
        "return json", "facts", purpose="report", json_mode=True
    )

    assert result == '{"summary":"요약"}'
    assert captured["response_format"] == {"type": "json_object"}
    await http_client.aclose()


@pytest.mark.asyncio
async def test_deepseek_missing_key_fails_without_network_call() -> None:
    settings = AISettings(database_url="postgresql+asyncpg://test")
    client = DeepSeekClient(settings)
    with pytest.raises(DeepSeekUnavailable, match="not configured"):
        await client.generate("system", "user", purpose="chat")
    await client.close()
