from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request

from app.ai.config import AISettings
from app.chatbot.schemas import ChatMessageRequest, ChatMessageResponse


router = APIRouter(tags=["chatbot"])


@router.post(
    "/v1/chat/messages",
    response_model=ChatMessageResponse,
    response_model_by_alias=True,
)
async def chat(
    payload: ChatMessageRequest,
    request: Request,
    x_internal_token: str | None = Header(default=None),
) -> ChatMessageResponse:
    settings: AISettings = request.app.state.ai_settings
    if settings.internal_api_token and x_internal_token != settings.internal_api_token:
        raise HTTPException(status_code=401, detail="invalid internal token")
    if not getattr(request.app.state, "ai_ready", False):
        raise HTTPException(status_code=503, detail="chatbot is not ready")
    return await request.app.state.chatbot_service.answer(payload)
