from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import text
from app.ai.config import AISettings
from app.chatbot.router import router as chatbot_router
from app.chatbot.runtime import AIRuntime, create_ai_runtime
from app.core.config import Settings, validate_bundle
from app.core.session_manager import SessionManager
from app.core.twin_redis import TwinRedisStore
from app.core.twin_rate_limit import TwinRateLimiter
from app.db import TwinRepository, create_database
from app.db.anomaly_persistence import AnomalyPersistence
from app.routers.current_stage import router
from app.routers.twins import router as twins_router
from app.services.current_stage_service import CurrentStageService
from app.services.cell_risk_gnn import CellRiskGNNAnalyzer
from app.services.thermal_inference import ThermalInferenceClient
from app.services.twin_service import TwinService

LOGGER = logging.getLogger("ev-ai-combined")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.twin_ready = False
    app.state.readiness_error = None
    app.state.ai_ready = False
    app.state.report_worker_enabled = False
    app.state.report_worker_running = False
    app.state.report_worker_error = None
    settings = Settings.load()
    app.state.settings = settings
    app.state.twin_rate_limiter = TwinRateLimiter()
    engine = None
    redis = None
    ai_runtime: AIRuntime | None = None
    report_worker_stop = None
    report_worker_task = None
    try:
        manifest = validate_bundle(settings)
        from hybrid_safety_supervisor import HybridSafetySupervisorV2
        def factory(): return HybridSafetySupervisorV2(settings.bundle_dir / "models" / "hybrid_v1", settings.bundle_dir / "config" / "safety_policy.v2.json")
        app.state.manifest = manifest
        current_stage_service = CurrentStageService(
            SessionManager(factory, settings.session_ttl_seconds, settings.max_sessions)
        )
        app.state.current_stage_service = current_stage_service
        cell_risk_gnn = CellRiskGNNAnalyzer.from_bundle(
            settings.cell_risk_gnn_dir
        )
        cell_risk_gnn.require_available()
        app.state.cell_risk_gnn = cell_risk_gnn
        app.state.cell_risk_gnn_ready = cell_risk_gnn.available
        engine, sessions = create_database(settings.database_url)
        redis = Redis.from_url(settings.redis_url, decode_responses=False)
        redis_store = TwinRedisStore(redis)
        if settings.redis_required:
            await redis_store.ping()
        app.state.database_engine = engine
        app.state.database_sessions = sessions
        anomaly_persistence = AnomalyPersistence(
            sessions,
            enqueue_report_jobs=settings.report_jobs_enabled,
        )
        app.state.anomaly_persistence = anomaly_persistence
        app.state.anomaly_persistence_enabled = settings.anomaly_persistence_enabled
        app.state.twin_redis = redis_store
        app.state.twin_service = TwinService(
            current_stage_service,
            redis_store,
            sessions,
            TwinRepository(),
            ThermalInferenceClient(
                settings.thermal_inference_url,
                settings.thermal_inference_token,
                settings.thermal_inference_timeout_seconds,
            ),
            anomaly_persistence=(
                anomaly_persistence
                if settings.anomaly_persistence_enabled
                else None
            ),
            cell_risk_analyzer=cell_risk_gnn,
        )
        if settings.twin_infra_required:
            await redis_store.ping()
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
                table = await connection.scalar(
                    text("SELECT to_regclass('public.twin_incidents')")
                )
                if table is None:
                    raise RuntimeError("twin database migration has not completed")
        if settings.anomaly_persistence_enabled:
            async with engine.connect() as connection:
                anomaly_logs = await connection.scalar(
                    text("SELECT to_regclass('public.\"ANOMALY_LOGS\"')")
                )
                twin_frames = await connection.scalar(
                    text("SELECT to_regclass('public.\"TWIN_FRAMES\"')")
                )
                if anomaly_logs is None or twin_frames is None:
                    raise RuntimeError(
                        "ANOMALY_LOGS and TWIN_FRAMES must exist for anomaly persistence"
                    )
        ai_settings = AISettings.load()
        app.state.ai_settings = ai_settings
        if ai_settings.embedded_ai_enabled:
            if ai_settings.database_url == settings.database_url:
                ai_runtime = create_ai_runtime(
                    ai_settings,
                    engine=engine,
                    sessions=sessions,
                )
            else:
                ai_runtime = create_ai_runtime(ai_settings)
            app.state.chatbot_service = ai_runtime.chatbot_service
            app.state.ai_ready = True
            app.state.report_worker_enabled = ai_settings.report_worker_enabled
            if ai_settings.report_worker_enabled:
                report_worker_stop = asyncio.Event()
                report_worker_task = asyncio.create_task(
                    ai_runtime.run_report_worker(report_worker_stop),
                    name="ai-report-worker",
                )
                app.state.report_worker_running = True

                def record_worker_result(task: asyncio.Task[None]) -> None:
                    app.state.report_worker_running = False
                    if task.cancelled():
                        return
                    error = task.exception()
                    if error is not None:
                        message = f"report worker stopped unexpectedly: {error}"
                        app.state.report_worker_error = str(error)
                        app.state.readiness_error = message
                        app.state.ready = False
                        LOGGER.error(
                            message,
                            exc_info=(type(error), error, error.__traceback__),
                        )

                report_worker_task.add_done_callback(record_worker_result)
        app.state.twin_ready = True
        app.state.ready = True
    except Exception as exc:
        LOGGER.exception("FastAPI startup failed; readiness disabled")
        app.state.readiness_error = str(exc)
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
                pass
        if ai_runtime is not None:
            await ai_runtime.close()
        if redis is not None:
            await redis.aclose()
        if engine is not None:
            await engine.dispose()

_settings = Settings.load()
app = FastAPI(title="EV Battery Safety Inference API", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    # Do not echo NaN/Infinity back into JSON; a standards-compliant 422 body is required.
    detail = [{"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})

app.include_router(router)
app.include_router(twins_router)
app.include_router(chatbot_router)
