from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import text
from app.core.config import Settings, validate_bundle
from app.core.session_manager import SessionManager
from app.core.twin_redis import TwinRedisStore
from app.db import TwinRepository, create_database
from app.db.anomaly_persistence import AnomalyPersistence
from app.routers.current_stage import router
from app.routers.twins import router as twins_router
from app.services.current_stage_service import CurrentStageService
from app.services.thermal_inference import ThermalInferenceClient
from app.services.twin_service import TwinService

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.twin_ready = False
    app.state.readiness_error = None
    settings = Settings.load()
    app.state.settings = settings
    engine = None
    redis = None
    try:
        manifest = validate_bundle(settings)
        from hybrid_safety_supervisor import HybridSafetySupervisorV2
        def factory(): return HybridSafetySupervisorV2(settings.bundle_dir / "models" / "hybrid_v1", settings.bundle_dir / "config" / "safety_policy.v2.json")
        app.state.manifest = manifest
        current_stage_service = CurrentStageService(
            SessionManager(factory, settings.session_ttl_seconds, settings.max_sessions)
        )
        app.state.current_stage_service = current_stage_service
        engine, sessions = create_database(settings.database_url)
        redis = Redis.from_url(settings.redis_url, decode_responses=False)
        redis_store = TwinRedisStore(redis)
        app.state.database_engine = engine
        app.state.database_sessions = sessions
        app.state.anomaly_persistence = AnomalyPersistence(sessions)
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
        app.state.twin_ready = True
        app.state.ready = True
    except Exception as exc:
        app.state.readiness_error = str(exc)
    try:
        yield
    finally:
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
