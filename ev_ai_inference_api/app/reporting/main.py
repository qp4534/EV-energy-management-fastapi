from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from typing import Protocol

from fastapi import FastAPI, Header, HTTPException, Request

from app.ai.config import AISettings
from app.db.session import create_database

from .repository import PostgresReportRepository
from .schemas import JobResponse, MonthlyJobRequest


class JobQueue(Protocol):
    async def enqueue_anomaly(self, anomaly_id: str): ...

    async def enqueue_monthly(self, target_month: date): ...


def create_report_job_app(queue: JobQueue | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        engine = None
        settings = AISettings.load()
        app.state.settings = settings
        if queue is not None:
            app.state.report_jobs = queue
        else:
            engine, sessions = create_database(settings.database_url)
            app.state.report_jobs = PostgresReportRepository(sessions)
        app.state.ready = True
        try:
            yield
        finally:
            if engine is not None:
                await engine.dispose()

    application = FastAPI(
        title="EV AI Report Job API",
        version="1.0.0",
        lifespan=lifespan,
    )

    def authorize(request: Request, token: str | None) -> None:
        settings: AISettings = request.app.state.settings
        if settings.internal_api_token and token != settings.internal_api_token:
            raise HTTPException(status_code=401, detail="invalid internal token")

    @application.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/readyz")
    async def readyz(request: Request) -> dict[str, str]:
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(status_code=503, detail="report job API is not ready")
        return {"status": "ok"}

    @application.post(
        "/internal/v1/report-jobs/anomalies/{anomaly_id}",
        response_model=JobResponse,
        status_code=202,
    )
    async def enqueue_anomaly(
        anomaly_id: str,
        request: Request,
        x_internal_token: str | None = Header(default=None),
    ) -> JobResponse:
        authorize(request, x_internal_token)
        try:
            job_id = await request.app.state.report_jobs.enqueue_anomaly(anomaly_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid anomalyId") from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JobResponse(job_id=str(job_id), status="PENDING")

    @application.post(
        "/internal/v1/report-jobs/monthly",
        response_model=JobResponse,
        status_code=202,
    )
    async def enqueue_monthly(
        payload: MonthlyJobRequest,
        request: Request,
        x_internal_token: str | None = Header(default=None),
    ) -> JobResponse:
        authorize(request, x_internal_token)
        target_month = date.fromisoformat(f"{payload.target_month}-01")
        try:
            job_id = await request.app.state.report_jobs.enqueue_monthly(target_month)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid targetMonth") from exc
        return JobResponse(job_id=str(job_id), status="PENDING")

    return application


app = create_report_job_app()
