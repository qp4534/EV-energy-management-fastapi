from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.core.config import Settings, validate_bundle
from app.core.session_manager import SessionManager
from app.routers.current_stage import router
from app.services.current_stage_service import CurrentStageService

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.readiness_error = None
    settings = Settings.load()
    try:
        manifest = validate_bundle(settings)
        from hybrid_safety_supervisor import HybridSafetySupervisorV2
        def factory(): return HybridSafetySupervisorV2(settings.bundle_dir / "models" / "hybrid_v1", settings.bundle_dir / "config" / "safety_policy.v2.json")
        app.state.manifest = manifest
        app.state.current_stage_service = CurrentStageService(SessionManager(factory, settings.session_ttl_seconds, settings.max_sessions))
        app.state.ready = True
    except Exception as exc:
        app.state.readiness_error = str(exc)
    yield

app = FastAPI(title="EV Battery Safety Inference API", version="1.0.0", lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    # Do not echo NaN/Infinity back into JSON; a standards-compliant 422 body is required.
    detail = [{"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})

app.include_router(router)
