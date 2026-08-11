from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Protocol
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")


class DataQueryKind(StrEnum):
    RISK_OVERVIEW = "RISK_OVERVIEW"
    ANOMALY_SUMMARY = "ANOMALY_SUMMARY"
    REPORT_JOB_STATUS = "REPORT_JOB_STATUS"
    LOW_SOH_BATTERIES = "LOW_SOH_BATTERIES"
    CHARGING_SUMMARY = "CHARGING_SUMMARY"


@dataclass(frozen=True)
class DataQuerySpec:
    kind: DataQueryKind
    period_label: str | None = None
    start_at: datetime | None = None
    end_at: datetime | None = None
    threshold: float | None = None
    limit: int | None = None

    def filters(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if self.period_label:
            values["period"] = self.period_label
        if self.start_at:
            values["startAt"] = self.start_at.isoformat()
        if self.end_at:
            values["endAtExclusive"] = self.end_at.isoformat()
        if self.threshold is not None:
            values["threshold"] = self.threshold
        if self.limit is not None:
            values["limit"] = self.limit
        return values


@dataclass(frozen=True)
class ChatDataResult:
    kind: DataQueryKind
    data: dict[str, Any]
    data_as_of: datetime
    source_tables: tuple[str, ...]
    filters: dict[str, Any]


class ChatDataProvider(Protocol):
    async def fetch(self, spec: DataQuerySpec) -> ChatDataResult:
        ...


class ActorRole(StrEnum):
    ADMIN = "ADMIN"
    OPERATOR = "OPERATOR"
    USER = "USER"
    UNKNOWN = "UNKNOWN"


def normalize_actor_role(value: str | None) -> ActorRole:
    normalized = (value or "").strip().lower()
    if normalized in {"관리자", "admin", "role_admin"}:
        return ActorRole.ADMIN
    if normalized in {"관제자", "controller", "operator", "role_controller"}:
        return ActorRole.OPERATOR
    if normalized in {"이용자", "사용자", "user", "role_user"}:
        return ActorRole.USER
    return ActorRole.UNKNOWN


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _month_window(year: int, month: int) -> tuple[datetime, datetime, str]:
    start = datetime(year, month, 1, tzinfo=KST)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=KST)
    else:
        end = datetime(year, month + 1, 1, tzinfo=KST)
    return start, end, f"{year:04d}-{month:02d}"


def _period(message: str, now: datetime) -> tuple[datetime, datetime, str]:
    explicit = re.search(r"(20\d{2})\s*년\s*(1[0-2]|0?[1-9])\s*월", message)
    if explicit:
        return _month_window(int(explicit.group(1)), int(explicit.group(2)))

    current = _aware(now).astimezone(KST)
    if any(term in message for term in ("지난달", "지난 달", "전월")):
        if current.month == 1:
            return _month_window(current.year - 1, 12)
        return _month_window(current.year, current.month - 1)
    return _month_window(current.year, current.month)


def _soh_threshold(message: str) -> float:
    matched = re.search(
        r"soh\s*(?:가|는|이)?\s*(\d{1,3}(?:\.\d+)?)\s*%?\s*(?:미만|이하)",
        message,
        flags=re.IGNORECASE,
    )
    if not matched:
        return 70.0
    return min(100.0, max(0.0, float(matched.group(1))))


def detect_data_query(
    message: str,
    *,
    now: datetime | None = None,
) -> DataQuerySpec | None:
    """Map supported project-data questions to an allow-listed query contract.

    This parser never returns SQL or accepts table/column names from the user.
    """

    normalized = " ".join(message.lower().split())
    current = _aware(now or datetime.now(timezone.utc))

    if "soh" in normalized and any(
        term in normalized for term in ("미만", "이하", "낮", "저하", "몇 개", "몇개")
    ):
        return DataQuerySpec(
            kind=DataQueryKind.LOW_SOH_BATTERIES,
            threshold=_soh_threshold(normalized),
            limit=10,
        )

    if "보고서" in normalized and any(
        term in normalized
        for term in ("작업", "실패", "생성 상태", "생성 현황", "잡", "job")
    ):
        return DataQuerySpec(
            kind=DataQueryKind.REPORT_JOB_STATUS,
            period_label="최근 30일",
            start_at=current - timedelta(days=30),
            end_at=current + timedelta(seconds=1),
            limit=10,
        )

    if "충전" in normalized and any(
        term in normalized
        for term in ("세션", "총 충전 시간", "충전시간", "충전 시간", "완료된 충전")
    ):
        start, end, label = _period(normalized, current)
        return DataQuerySpec(
            kind=DataQueryKind.CHARGING_SUMMARY,
            period_label=label,
            start_at=start,
            end_at=end,
        )

    if any(term in normalized for term in ("이상", "화재")) and any(
        term in normalized
        for term in ("발생 건수", "발생건수", "몇 건", "몇건", "유형별", "현황", "통계")
    ):
        start, end, label = _period(normalized, current)
        return DataQuerySpec(
            kind=DataQueryKind.ANOMALY_SUMMARY,
            period_label=label,
            start_at=start,
            end_at=end,
        )

    risk_subject = any(
        term in normalized
        for term in ("위험등급", "위험 등급", "위험 차량", "위험차량")
    )
    risk_aggregate = any(
        term in normalized
        for term in ("차량 수", "차량수", "몇 대", "몇대", "분포", "현황", "전체")
    )
    if risk_subject and risk_aggregate:
        return DataQuerySpec(kind=DataQueryKind.RISK_OVERVIEW)

    return None
