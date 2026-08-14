from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy.exc import SQLAlchemyError
from app.core.session_manager import TimestampConflict
from app.schemas.current_stage import SampleRequest

router = APIRouter(tags=["current-stage"])

@router.get("/healthz")
async def healthz() -> dict[str, str]: return {"status": "ok"}

@router.get("/readyz")
async def readyz(request: Request):
    state = request.app.state
    if not getattr(state, "ready", False):
        raise HTTPException(status_code=503, detail={"status": "not_ready", "reason": getattr(state, "readiness_error", "bundle not loaded")})
    return {"status": "ok"}

@router.get("/v1/model-info")
async def model_info(request: Request):
    manifest = request.app.state.manifest
    return {"package_version": manifest["package_version"], "model_status": manifest["status"], "class_names": ["normal", "caution", "warning", "emergency"], "sampling_interval_seconds": manifest["sampling_interval_seconds"], "inference_interval_seconds": manifest["inference_interval_seconds"], "routing": manifest["routing"], "required_signals": manifest["required_causal_signals"], "description": "현재 및 과거 BMS 신호의 4단계 현재 상태 분류 모델입니다.", "not_a_180s_onset_model": "향후 180초 열폭주 진입 확률 모델이 아닙니다."}

@router.post("/v1/vehicles/{vehicle_id}/samples")
async def sample(vehicle_id: str, payload: SampleRequest, request: Request):
    if not request.app.state.ready: raise HTTPException(status_code=503, detail="model bundle is not ready")
    try:
        result = await request.app.state.current_stage_service.evaluate(vehicle_id, payload)
        if getattr(request.app.state, "anomaly_persistence_enabled", False):
            anomaly_id = await request.app.state.anomaly_persistence.persist_if_anomalous(
                vehicle_id, payload, result
            )
            if anomaly_id is not None:
                result = result.model_copy(update={"anomaly_id": anomaly_id})
        return result
    except TimestampConflict as exc: raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="anomaly persistence unavailable") from exc
    except Exception:
        raise HTTPException(status_code=500, detail="internal inference error")

@router.post("/v1/vehicles/{vehicle_id}/reset")
async def reset(vehicle_id: str, request: Request): return await request.app.state.current_stage_service.reset(vehicle_id)

@router.delete("/v1/vehicles/{vehicle_id}/session")
async def delete_session(vehicle_id: str, request: Request, response: Response):
    deleted = await request.app.state.current_stage_service.delete(vehicle_id)
    if not deleted["deleted"]: response.status_code = status.HTTP_404_NOT_FOUND
    return deleted
