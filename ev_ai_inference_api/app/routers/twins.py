from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

from app.core.session_manager import TimestampConflict
from app.schemas.twins import (
    IncidentListResponse,
    RiskVehicleListResponse,
    TwinFrame,
    TwinHistoryResponse,
    TwinSampleRequest,
)
from app.services.twin_service import (
    IncidentNotFound,
    InvalidVehicleId,
    TwinSequenceConflict,
)


router = APIRouter(prefix="/api/v1/twins", tags=["digital-twins"])


def _service(request: Request):
    service = getattr(request.app.state, "twin_service", None)
    if service is None or not getattr(request.app.state, "twin_ready", False):
        raise HTTPException(status_code=503, detail="twin infrastructure is not ready")
    return service


@router.post(
    "/vehicles/{vehicle_id}/samples",
    response_model=TwinFrame,
)
async def create_sample(
    vehicle_id: str, payload: TwinSampleRequest, request: Request
) -> TwinFrame:
    try:
        return await _service(request).evaluate(vehicle_id, payload)
    except (TimestampConflict, TwinSequenceConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (InvalidVehicleId, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RedisError, SQLAlchemyError, OSError) as exc:
        raise HTTPException(status_code=503, detail="twin infrastructure unavailable") from exc


@router.post(
    "/vehicles/{vehicle_id}/observations",
    response_model=TwinFrame,
)
async def create_observation(
    vehicle_id: str,
    request: Request,
    sample_json: str = Form(...),
    thermal_image: UploadFile = File(...),
) -> TwinFrame:
    """Evaluate one synchronized BMS sample and thermal image."""

    try:
        payload = TwinSampleRequest.model_validate_json(sample_json)
        if thermal_image.content_type not in {"image/png", "image/jpeg"}:
            raise ValueError("thermal_image must be PNG or JPEG")
        image_bytes = await thermal_image.read()
        if not image_bytes or len(image_bytes) > 2_000_000:
            raise ValueError("thermal_image must be between 1 byte and 2 MB")
        return await _service(request).evaluate_observation(
            vehicle_id, payload, image_bytes
        )
    except (TimestampConflict, TwinSequenceConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (InvalidVehicleId, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RedisError, SQLAlchemyError, OSError) as exc:
        raise HTTPException(status_code=503, detail="twin infrastructure unavailable") from exc


@router.get("/risk-vehicles", response_model=RiskVehicleListResponse)
async def risk_vehicles(request: Request) -> RiskVehicleListResponse:
    try:
        return await _service(request).risk_vehicles()
    except (RedisError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Redis unavailable") from exc


@router.get(
    "/vehicles/{vehicle_id}/incidents",
    response_model=IncidentListResponse,
)
async def incidents(vehicle_id: str, request: Request) -> IncidentListResponse:
    try:
        return await _service(request).incidents(vehicle_id)
    except InvalidVehicleId as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (SQLAlchemyError, OSError) as exc:
        raise HTTPException(status_code=503, detail="PostgreSQL unavailable") from exc


@router.get(
    "/vehicles/{vehicle_id}/incidents/latest/history",
    response_model=TwinHistoryResponse,
)
async def latest_history(
    vehicle_id: str,
    request: Request,
    resolution_seconds: int = Query(default=30, ge=1, le=3_600),
) -> TwinHistoryResponse:
    try:
        return await _service(request).latest_history(vehicle_id, resolution_seconds)
    except InvalidVehicleId as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IncidentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (SQLAlchemyError, OSError) as exc:
        raise HTTPException(status_code=503, detail="PostgreSQL unavailable") from exc


@router.get(
    "/vehicles/{vehicle_id}/incidents/{incident_id}/history",
    response_model=TwinHistoryResponse,
)
async def incident_history(
    vehicle_id: str,
    incident_id: str,
    request: Request,
    resolution_seconds: int = Query(default=30, ge=1, le=3_600),
) -> TwinHistoryResponse:
    try:
        return await _service(request).incident_history(
            vehicle_id, incident_id, resolution_seconds
        )
    except InvalidVehicleId as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IncidentNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (SQLAlchemyError, OSError) as exc:
        raise HTTPException(status_code=503, detail="PostgreSQL unavailable") from exc


@router.websocket("/vehicles/{vehicle_id}/live")
async def live(vehicle_id: str, websocket: WebSocket) -> None:
    service = getattr(websocket.app.state, "twin_service", None)
    if service is None or not getattr(websocket.app.state, "twin_ready", False):
        await websocket.close(code=1013, reason="twin infrastructure is not ready")
        return
    await websocket.accept()
    try:
        async for frame in service.live_frames(vehicle_id):
            await websocket.send_text(frame.model_dump_json())
    except InvalidVehicleId:
        await websocket.close(code=1008, reason="invalid vehicle_id")
    except WebSocketDisconnect:
        return
    except (RedisError, OSError):
        await websocket.close(code=1013, reason="Redis unavailable")
