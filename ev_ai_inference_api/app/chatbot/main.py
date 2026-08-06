from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request

from app.ai.config import AISettings
from app.ai.deepseek import DeepSeekClient
from app.chatbot.schemas import ChatMessageRequest, ChatMessageResponse
from app.chatbot.service import ChatbotService
from app.chatbot.supervisor import ChatSupervisor
from app.chatbot.vehicle_client import InferenceVehicleStateClient
from app.db.session import create_database
from app.rag.embedding import SentenceTransformerEmbedder
from app.rag.repository import PostgresRagRepository


def create_chatbot_app(service: ChatbotService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        deepseek = None
        vehicle_client = None
        engine = None
        if service is not None:
            app.state.chatbot_service = service
            app.state.settings = service.settings
            app.state.ready = True
            yield
            return

        settings = AISettings.load()
        app.state.settings = settings
        engine, sessions = create_database(settings.database_url)
        embedder = SentenceTransformerEmbedder(
            settings.embedding_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
        )
        rag = PostgresRagRepository(sessions, embedder, settings)
        deepseek = DeepSeekClient(settings)
        vehicle_client = InferenceVehicleStateClient(
            settings.inference_base_url,
            timeout_seconds=settings.vehicle_state_timeout_seconds,
            internal_token=settings.internal_api_token,
        )
        app.state.chatbot_service = ChatbotService(
            ChatSupervisor(), rag, deepseek, vehicle_client, settings
        )
        app.state.ready = True
        try:
            yield
        finally:
            if vehicle_client is not None:
                await vehicle_client.close()
            if deepseek is not None:
                await deepseek.close()
            if engine is not None:
                await engine.dispose()

    application = FastAPI(
        title="EV User Chatbot API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/readyz")
    async def readyz(request: Request) -> dict[str, object]:
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(status_code=503, detail="chatbot is not ready")
        settings: AISettings = request.app.state.settings
        return {
            "status": "ok",
            "deepseekConfigured": settings.deepseek_configured,
        }

    @application.post(
        "/v1/chat/messages",
        response_model=ChatMessageResponse,
        response_model_by_alias=True,
    )
    async def chat(
        payload: ChatMessageRequest,
        request: Request,
        x_internal_token: str | None = Header(default=None),
    ) -> ChatMessageResponse:
        settings: AISettings = request.app.state.settings
        if settings.internal_api_token and x_internal_token != settings.internal_api_token:
            raise HTTPException(status_code=401, detail="invalid internal token")
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(status_code=503, detail="chatbot is not ready")
        return await request.app.state.chatbot_service.answer(payload)

    return application


app = create_chatbot_app()
