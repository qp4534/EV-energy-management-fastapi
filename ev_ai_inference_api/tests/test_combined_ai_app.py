import asyncio

from fastapi.testclient import TestClient

from app import main as main_module
from app.chatbot.schemas import ChatMessageResponse


class FakeChatbotService:
    async def answer(self, request):
        return ChatMessageResponse(
            answer=f"combined: {request.message}",
            route="RAG",
            safety_level="NORMAL",
        )


class FakeAIRuntime:
    def __init__(self, lifecycle: list[str]) -> None:
        self.lifecycle = lifecycle
        self.chatbot_service = FakeChatbotService()

    async def run_report_worker(self, stop_event: asyncio.Event) -> None:
        self.lifecycle.append("worker-started")
        await stop_event.wait()
        self.lifecycle.append("worker-stopped")

    async def close(self) -> None:
        self.lifecycle.append("runtime-closed")


def test_default_app_runs_inference_chatbot_and_report_worker_together(
    monkeypatch,
) -> None:
    lifecycle: list[str] = []
    monkeypatch.setenv("EMBEDDED_AI_ENABLED", "true")
    monkeypatch.setenv("REPORT_WORKER_ENABLED", "true")
    monkeypatch.setattr(
        main_module,
        "create_ai_runtime",
        lambda settings, **kwargs: FakeAIRuntime(lifecycle),
    )

    with TestClient(main_module.app) as client:
        assert client.get("/readyz").status_code == 200
        assert client.get("/v1/model-info").status_code == 200
        response = client.post(
            "/v1/chat/messages",
            json={"message": "질문", "vehicleId": "car-1"},
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "combined: 질문"
        assert lifecycle == ["worker-started"]

    assert lifecycle == [
        "worker-started",
        "worker-stopped",
        "runtime-closed",
    ]
