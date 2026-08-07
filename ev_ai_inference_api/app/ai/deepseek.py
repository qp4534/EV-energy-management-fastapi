from __future__ import annotations

from typing import Literal

import httpx

from .config import AISettings


class DeepSeekUnavailable(RuntimeError):
    """Raised when a model response cannot safely be used."""


class DeepSeekClient:
    def __init__(
        self,
        settings: AISettings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=settings.deepseek_timeout_seconds
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        purpose: Literal["chat", "report"],
        json_mode: bool = False,
    ) -> str:
        if not self.settings.deepseek_api_key:
            raise DeepSeekUnavailable("DeepSeek API key is not configured")

        is_report = purpose == "report"
        body: dict[str, object] = {
            "model": (
                self.settings.deepseek_report_model
                if is_report
                else self.settings.deepseek_chat_model
            ),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "thinking": {
                "type": (
                    "enabled"
                    if is_report and self.settings.deepseek_report_thinking
                    else "disabled"
                )
            },
            "max_tokens": (
                self.settings.deepseek_report_max_tokens
                if is_report
                else self.settings.deepseek_chat_max_tokens
            ),
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        try:
            response = await self._client.post(
                f"{self.settings.deepseek_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            if choice.get("finish_reason") == "length":
                raise DeepSeekUnavailable("DeepSeek response was truncated")
            content = choice["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise DeepSeekUnavailable("DeepSeek returned an empty response")
            return content.strip()
        except DeepSeekUnavailable:
            raise
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise DeepSeekUnavailable("DeepSeek request failed") from exc
