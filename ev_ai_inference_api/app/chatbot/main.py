from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.ai.config import AISettings
from app.chatbot.router import router as chatbot_router
from app.chatbot.runtime import AIRuntime, create_ai_runtime
from app.chatbot.service import ChatbotService


LOGGER = logging.getLogger("ev-ai-chatbot")
ReportWorkerRunner = Callable[[asyncio.Event], Awaitable[None]]


def create_chatbot_app(
    service: ChatbotService | None = None,
    report_worker_runner: ReportWorkerRunner | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        app.state.report_worker_enabled = False
        app.state.report_worker_running = False
        app.state.report_worker_error = None
        app.state.ai_ready = False
        runtime: AIRuntime | None = None
        report_worker_stop = None
        report_worker_task = None

        if service is not None:
            app.state.chatbot_service = service
            settings = service.settings
        else:
            settings = AISettings.load()
            runtime = create_ai_runtime(settings)
            app.state.chatbot_service = runtime.chatbot_service

        app.state.settings = settings
        app.state.ai_settings = settings
        app.state.ai_ready = True

        app.state.report_worker_enabled = settings.report_worker_enabled
        if settings.report_worker_enabled:
            report_worker_stop = asyncio.Event()
            if report_worker_runner is not None:
                report_worker_coroutine = report_worker_runner(report_worker_stop)
            else:
                if runtime is None:
                    raise RuntimeError(
                        "an injected chatbot service requires an injected report worker"
                    )
                report_worker_coroutine = runtime.run_report_worker(
                    report_worker_stop
                )
            report_worker_task = asyncio.create_task(
                report_worker_coroutine,
                name="ai-report-worker",
            )
            app.state.report_worker_running = True

            def record_worker_result(task: asyncio.Task[None]) -> None:
                app.state.report_worker_running = False
                if task.cancelled():
                    return
                error = task.exception()
                if error is not None:
                    app.state.report_worker_error = str(error)
                    LOGGER.error(
                        "report worker stopped unexpectedly",
                        exc_info=(type(error), error, error.__traceback__),
                    )

            report_worker_task.add_done_callback(record_worker_result)

        app.state.ready = True
        try:
            yield
        finally:
            if report_worker_stop is not None:
                report_worker_stop.set()
            if report_worker_task is not None:
                try:
                    await report_worker_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    # The completion callback already captured and logged the failure.
                    pass
            if runtime is not None:
                await runtime.close()

    application = FastAPI(
        title="EV User Chatbot API",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.include_router(chatbot_router)

    @application.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/readyz")
    async def readyz(request: Request) -> dict[str, object]:
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(status_code=503, detail="chatbot is not ready")
        if (
            getattr(request.app.state, "report_worker_enabled", False)
            and not getattr(request.app.state, "report_worker_running", False)
        ):
            raise HTTPException(status_code=503, detail="report worker is not running")
        settings: AISettings = request.app.state.settings
        return {
            "status": "ok",
            "deepseekConfigured": settings.deepseek_configured,
            "reportWorkerEnabled": request.app.state.report_worker_enabled,
            "reportWorkerRunning": request.app.state.report_worker_running,
        }

    return application


app = create_chatbot_app()
