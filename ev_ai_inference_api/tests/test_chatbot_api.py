from fastapi.testclient import TestClient

from app.ai.config import AISettings
from app.chatbot.main import create_chatbot_app
from app.chatbot.schemas import ChatMessageResponse


class FakeChatbotService:
    settings = AISettings(database_url="postgresql+asyncpg://test")

    async def answer(self, request):
        return ChatMessageResponse(
            answer=f"echo: {request.message}",
            route="RAG",
            safety_level="NORMAL",
            fallback_used=False,
        )


def test_chatbot_api_uses_spring_facing_camel_case_contract() -> None:
    app = create_chatbot_app(FakeChatbotService())  # type: ignore[arg-type]
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/messages",
            json={"message": "질문", "vehicleId": "car-1"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "echo: 질문",
        "route": "RAG",
        "safetyLevel": "NORMAL",
        "dataAsOf": None,
        "sources": [],
        "missingFields": [],
        "fallbackUsed": False,
        "metadata": {},
    }
