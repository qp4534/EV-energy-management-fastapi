from __future__ import annotations

from enum import StrEnum


class ChatRoute(StrEnum):
    EMERGENCY = "EMERGENCY"
    VEHICLE_STATUS = "VEHICLE_STATUS"
    ADMIN_DATA = "ADMIN_DATA"
    OPERATOR_DATA = "OPERATOR_DATA"
    DATA_QUERY = "DATA_QUERY"
    LEGAL = "LEGAL"
    RAG = "RAG"
    GENERAL = "GENERAL"


_EMERGENCY_TERMS = (
    "화재",
    "불이 났",
    "불났",
    "연기",
    "폭발",
    "불꽃",
    "스파크",
    "감전",
    "타는 냄새",
    "탄 냄새",
    "사람이 다쳤",
    "과열",
)
_LEGAL_TERMS = (
    "법령",
    "법적",
    "법률",
    "규정",
    "고시",
    "조항",
    "한국전기설비규정",
    "kec",
)
_CURRENT_MARKERS = ("지금", "현재", "실시간", "내 차", "내차", "우리 차")
_VEHICLE_SIGNAL_TERMS = (
    "상태",
    "배터리",
    "온도",
    "전압",
    "전류",
    "soc",
    "위험",
    "경고",
    "충전",
)
_GENERAL_ONLY = (
    "안녕",
    "고마워",
    "감사",
    "너는 누구",
    "무엇을 할 수",
    "뭘 할 수",
)


class ChatSupervisor:
    """Deterministic safety-first router; it does not call an LLM."""

    def classify(self, message: str) -> ChatRoute:
        normalized = " ".join(message.lower().split())
        if any(term in normalized for term in _EMERGENCY_TERMS):
            return ChatRoute.EMERGENCY
        if any(term in normalized for term in _LEGAL_TERMS):
            return ChatRoute.LEGAL
        if any(marker in normalized for marker in _CURRENT_MARKERS) and any(
            term in normalized for term in _VEHICLE_SIGNAL_TERMS
        ):
            return ChatRoute.VEHICLE_STATUS
        if any(term in normalized for term in _GENERAL_ONLY):
            return ChatRoute.GENERAL
        return ChatRoute.RAG
