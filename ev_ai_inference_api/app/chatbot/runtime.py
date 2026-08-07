from __future__ import annotations

import asyncio
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.ai.config import AISettings
from app.ai.deepseek import DeepSeekClient
from app.chatbot.service import ChatbotService
from app.chatbot.supervisor import ChatSupervisor
from app.chatbot.vehicle_client import InferenceVehicleStateClient
from app.db.session import create_database
from app.rag.embedding import SentenceTransformerEmbedder
from app.rag.repository import PostgresRagRepository
from app.reporting.repository import PostgresReportRepository
from app.reporting.service import ReportGenerationService
from app.reporting.worker import run_loop


@dataclass
class AIRuntime:
    settings: AISettings
    engine: AsyncEngine
    owns_engine: bool
    chatbot_service: ChatbotService
    report_repository: PostgresReportRepository
    report_service: ReportGenerationService
    deepseek: DeepSeekClient
    vehicle_client: InferenceVehicleStateClient

    async def run_report_worker(self, stop_event: asyncio.Event) -> None:
        await run_loop(
            self.report_repository,
            self.report_service,
            self.settings,
            stop_event,
        )

    async def close(self) -> None:
        await self.vehicle_client.close()
        await self.deepseek.close()
        if self.owns_engine:
            await self.engine.dispose()


def create_ai_runtime(
    settings: AISettings,
    *,
    engine: AsyncEngine | None = None,
    sessions: async_sessionmaker[AsyncSession] | None = None,
) -> AIRuntime:
    if (engine is None) != (sessions is None):
        raise ValueError("engine and sessions must be provided together")
    owns_engine = engine is None
    if engine is None or sessions is None:
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
    chatbot_service = ChatbotService(
        ChatSupervisor(),
        rag,
        deepseek,
        vehicle_client,
        settings,
    )
    report_repository = PostgresReportRepository(sessions)
    report_service = ReportGenerationService(report_repository, rag, deepseek)
    return AIRuntime(
        settings=settings,
        engine=engine,
        owns_engine=owns_engine,
        chatbot_service=chatbot_service,
        report_repository=report_repository,
        report_service=report_service,
        deepseek=deepseek,
        vehicle_client=vehicle_client,
    )
