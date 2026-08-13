from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request, WebSocket

from app.core.config import Settings


SERVICE_TOKEN_HEADER = "X-Twin-Service-Token"
WEBSOCKET_PROTOCOL = "twin-v1"
WEBSOCKET_TOKEN_PREFIX = "auth."


@dataclass(frozen=True)
class TwinTicket:
    subject: str
    vehicle_id: str
    scope: str
    expires_at: int


def _settings(container: Any) -> Settings:
    settings = getattr(container.app.state, "settings", None)
    if not isinstance(settings, Settings):
        raise HTTPException(status_code=503, detail="Twin authorization is unavailable")
    return settings


def _service_token_matches(settings: Settings, supplied: str | None) -> bool:
    return bool(
        supplied
        and settings.twin_service_token
        and hmac.compare_digest(supplied, settings.twin_service_token)
    )


def require_twin_service(request: Request) -> None:
    settings = _settings(request)
    if not settings.twin_auth_required:
        return
    if not _service_token_matches(
        settings, request.headers.get(SERVICE_TOKEN_HEADER)
    ):
        raise HTTPException(status_code=401, detail="Twin service authentication required")


def _decode_ticket(settings: Settings, encoded: str) -> TwinTicket:
    try:
        header_segment, payload_segment, signature_segment = encoded.split(".")
        header = json.loads(_base64url_decode(header_segment))
        claims = json.loads(_base64url_decode(payload_segment))
        if header.get("alg") != "HS256":
            raise ValueError("unsupported signing algorithm")
        signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
        signing_key = hashlib.sha256(
            settings.twin_ticket_secret.encode("utf-8")
        ).digest()
        expected_signature = hmac.new(
            signing_key, signing_input, hashlib.sha256
        ).digest()
        supplied_signature = _base64url_decode(signature_segment)
        if not hmac.compare_digest(expected_signature, supplied_signature):
            raise ValueError("signature mismatch")
        required = {"sub", "vehicle_id", "scope", "iat", "exp", "aud", "iss"}
        if not required.issubset(claims):
            raise ValueError("required claims missing")
        now = int(time.time())
        if int(claims["exp"]) <= now or int(claims["iat"]) > now + 5:
            raise ValueError("ticket expired or issued in the future")
        audience = claims["aud"]
        audiences = {audience} if isinstance(audience, str) else set(audience)
        if settings.twin_ticket_audience not in audiences:
            raise ValueError("audience mismatch")
        if claims["iss"] != settings.twin_ticket_issuer:
            raise ValueError("issuer mismatch")
        return TwinTicket(
            subject=str(claims["sub"]),
            vehicle_id=str(claims["vehicle_id"]),
            scope=str(claims["scope"]),
            expires_at=int(claims["exp"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, binascii.Error) as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired Twin access ticket") from exc


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def require_twin_read(request: Request, vehicle_id: str) -> TwinTicket | None:
    settings = _settings(request)
    if not settings.twin_auth_required:
        return None
    if _service_token_matches(settings, request.headers.get(SERVICE_TOKEN_HEADER)):
        return None
    encoded = _bearer_token(request)
    if encoded is None:
        raise HTTPException(status_code=401, detail="Twin access ticket required")
    ticket = _decode_ticket(settings, encoded)
    if ticket.scope != "twin:read" or ticket.vehicle_id != vehicle_id:
        raise HTTPException(status_code=403, detail="Twin vehicle access denied")
    return ticket


def require_twin_websocket(websocket: WebSocket, vehicle_id: str) -> TwinTicket | None:
    settings = _settings(websocket)
    if not settings.twin_auth_required:
        return None
    origin = websocket.headers.get("origin")
    if origin not in settings.cors_origins:
        raise HTTPException(status_code=403, detail="Twin WebSocket origin denied")
    protocols = [
        value.strip()
        for value in websocket.headers.get("sec-websocket-protocol", "").split(",")
        if value.strip()
    ]
    encoded = next(
        (
            value[len(WEBSOCKET_TOKEN_PREFIX) :]
            for value in protocols
            if value.startswith(WEBSOCKET_TOKEN_PREFIX)
        ),
        None,
    )
    if WEBSOCKET_PROTOCOL not in protocols or not encoded:
        raise HTTPException(status_code=401, detail="Twin WebSocket access ticket required")
    ticket = _decode_ticket(settings, encoded)
    if ticket.scope != "twin:read" or ticket.vehicle_id != vehicle_id:
        raise HTTPException(status_code=403, detail="Twin vehicle access denied")
    return ticket
