import asyncio

from fastapi.testclient import TestClient

from app.ai.config import AISettings
from app.chatbot.main import create_chatbot_app
from app.chatbot.schemas import ChatMessageResponse


class FakeChatbotService:
    settings = AISettings(
        database_url="postgresql+asyncpg://test",
        report_worker_enabled=True,
    )

    async def answer(self, request):
        return ChatMessageResponse(
            answer=request.message,
            route="RAG",
            safety_level="NORMAL",
        )


def test_chatbot_lifespan_starts_and_stops_report_worker() -> None:
    lifecycle: list[str] = []

    async def fake_report_worker(stop_event: asyncio.Event) -> None:
        lifecycle.append("started")
        await stop_event.wait()
        lifecycle.append("stopped")

    app = create_chatbot_app(
        FakeChatbotService(),  # type: ignore[arg-type]
        report_worker_runner=fake_report_worker,
    )

    with TestClient(app) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["reportWorkerEnabled"] is True
        assert ready.json()["reportWorkerRunning"] is True
        assert lifecycle == ["started"]

    assert lifecycle == ["started", "stopped"]


def test_ready_fails_when_embedded_report_worker_crashes() -> None:
    crashed = asyncio.Event()

    async def failing_report_worker(_: asyncio.Event) -> None:
        crashed.set()
        raise RuntimeError("worker failed")

    app = create_chatbot_app(
        FakeChatbotService(),  # type: ignore[arg-type]
        report_worker_runner=failing_report_worker,
    )

    with TestClient(app) as client:
        assert crashed.is_set()
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["detail"] == "report worker is not running"
