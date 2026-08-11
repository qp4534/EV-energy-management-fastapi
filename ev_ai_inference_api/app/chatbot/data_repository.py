from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .data_queries import ChatDataResult, DataQueryKind, DataQuerySpec


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _iso(value: Any) -> str | None:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return None


def _as_datetime(value: Any, fallback: datetime) -> datetime:
    if not isinstance(value, datetime):
        return fallback
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class PostgresChatDataRepository:
    """Read-only, allow-listed project data queries for the chatbot."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.sessions = sessions
        self.now = now or (lambda: datetime.now(timezone.utc))

    async def fetch(self, spec: DataQuerySpec) -> ChatDataResult:
        handlers = {
            DataQueryKind.RISK_OVERVIEW: self._risk_overview,
            DataQueryKind.ANOMALY_SUMMARY: self._anomaly_summary,
            DataQueryKind.REPORT_JOB_STATUS: self._report_job_status,
            DataQueryKind.LOW_SOH_BATTERIES: self._low_soh_batteries,
            DataQueryKind.CHARGING_SUMMARY: self._charging_summary,
        }
        return await handlers[spec.kind](spec)

    async def _risk_overview(self, spec: DataQuerySpec) -> ChatDataResult:
        queried_at = self.now()
        stale_before = queried_at - timedelta(minutes=5)
        future_cutoff = queried_at + timedelta(seconds=30)
        statement = text(
            """
            WITH latest AS (
                SELECT DISTINCT ON (car_id)
                    car_id, final_risk_level, observed_at
                FROM public."TWIN_FRAMES"
                WHERE final_risk_level IS NOT NULL
                  AND observed_at <= :future_cutoff
                ORDER BY car_id, observed_at DESC, frame_id DESC
            )
            SELECT
                COUNT(c.car_id) AS total_vehicles,
                COUNT(latest.car_id) AS vehicles_with_twin,
                COUNT(*) FILTER (
                    WHERE latest.observed_at >= :stale_before
                ) AS fresh_vehicles,
                COUNT(*) FILTER (
                    WHERE latest.observed_at >= :stale_before
                      AND latest.final_risk_level = 0
                ) AS normal_count,
                COUNT(*) FILTER (
                    WHERE latest.observed_at >= :stale_before
                      AND latest.final_risk_level = 1
                ) AS caution_count,
                COUNT(*) FILTER (
                    WHERE latest.observed_at >= :stale_before
                      AND latest.final_risk_level = 2
                ) AS warning_count,
                COUNT(*) FILTER (
                    WHERE latest.observed_at >= :stale_before
                      AND latest.final_risk_level = 3
                ) AS emergency_count,
                COUNT(*) FILTER (
                    WHERE latest.car_id IS NOT NULL
                      AND latest.observed_at < :stale_before
                ) AS stale_count,
                MAX(latest.observed_at) AS latest_observed_at
            FROM public."CAR" c
            LEFT JOIN latest ON latest.car_id = c.car_id
            """
        )
        async with self.sessions() as session:
            row = (await session.execute(
                statement,
                {
                    "stale_before": stale_before,
                    "future_cutoff": future_cutoff,
                },
            )).mappings().one()
        total = int(row["total_vehicles"] or 0)
        measured = int(row["vehicles_with_twin"] or 0)
        fresh = int(row["fresh_vehicles"] or 0)
        data = {
            "totalVehicles": total,
            "vehiclesWithTwin": measured,
            "freshVehicles": fresh,
            "normal": int(row["normal_count"] or 0),
            "caution": int(row["caution_count"] or 0),
            "warning": int(row["warning_count"] or 0),
            "emergency": int(row["emergency_count"] or 0),
            "unknown": max(0, total - fresh),
            "staleVehicles": int(row["stale_count"] or 0),
            "staleAfterMinutes": 5,
        }
        return ChatDataResult(
            kind=spec.kind,
            data=data,
            data_as_of=_as_datetime(row["latest_observed_at"], queried_at),
            source_tables=("CAR", "TWIN_FRAMES"),
            filters=spec.filters(),
        )

    async def _anomaly_summary(self, spec: DataQuerySpec) -> ChatDataResult:
        if spec.start_at is None or spec.end_at is None:
            raise ValueError("anomaly summary requires a period")
        summary_sql = text(
            """
            SELECT
                COUNT(*) AS total_count,
                COUNT(*) FILTER (WHERE risk_level = '정상') AS normal_count,
                COUNT(*) FILTER (WHERE risk_level = '주의') AS caution_count,
                COUNT(*) FILTER (WHERE risk_level = '경고') AS warning_count,
                COUNT(*) FILTER (WHERE risk_level = '긴급') AS emergency_count,
                COUNT(DISTINCT car_id) AS affected_vehicles,
                MAX(detected_at) AS latest_detected_at
            FROM public."ANOMALY_LOGS"
            WHERE detected_at >= :start_at AND detected_at < :end_at
            """
        )
        types_sql = text(
            """
            SELECT abnormal_type, COUNT(*) AS event_count
            FROM public."ANOMALY_LOGS"
            WHERE detected_at >= :start_at AND detected_at < :end_at
            GROUP BY abnormal_type
            ORDER BY event_count DESC, abnormal_type
            LIMIT 10
            """
        )
        params = {"start_at": spec.start_at, "end_at": spec.end_at}
        async with self.sessions() as session:
            summary = (await session.execute(summary_sql, params)).mappings().one()
            type_rows = (await session.execute(types_sql, params)).mappings().all()
        data = {
            "totalEvents": int(summary["total_count"] or 0),
            "affectedVehicles": int(summary["affected_vehicles"] or 0),
            "byRiskLevel": {
                "정상": int(summary["normal_count"] or 0),
                "주의": int(summary["caution_count"] or 0),
                "경고": int(summary["warning_count"] or 0),
                "긴급": int(summary["emergency_count"] or 0),
            },
            "byType": [
                {
                    "type": row["abnormal_type"],
                    "count": int(row["event_count"] or 0),
                }
                for row in type_rows
            ],
        }
        return ChatDataResult(
            kind=spec.kind,
            data=data,
            data_as_of=_as_datetime(summary["latest_detected_at"], self.now()),
            source_tables=("ANOMALY_LOGS",),
            filters=spec.filters(),
        )

    async def _report_job_status(self, spec: DataQuerySpec) -> ChatDataResult:
        if spec.start_at is None or spec.end_at is None:
            raise ValueError("report job status requires a period")
        counts_sql = text(
            """
            SELECT status, COUNT(*) AS job_count, MAX(updated_at) AS latest_updated_at
            FROM ai_report_jobs
            WHERE updated_at >= :start_at AND updated_at < :end_at
            GROUP BY status
            ORDER BY status
            """
        )
        failures_sql = text(
            """
            SELECT job_type, retry_count, updated_at
            FROM ai_report_jobs
            WHERE status = 'FAILED'
              AND updated_at >= :start_at AND updated_at < :end_at
            ORDER BY updated_at DESC, job_id
            LIMIT :limit
            """
        )
        params = {
            "start_at": spec.start_at,
            "end_at": spec.end_at,
            "limit": spec.limit or 10,
        }
        async with self.sessions() as session:
            count_rows = (await session.execute(counts_sql, params)).mappings().all()
            failures = (await session.execute(failures_sql, params)).mappings().all()
        counts = {str(row["status"]): int(row["job_count"] or 0) for row in count_rows}
        latest_values = [row["latest_updated_at"] for row in count_rows]
        latest = max(
            (value for value in latest_values if isinstance(value, datetime)),
            default=self.now(),
        )
        data = {
            "totalJobs": sum(counts.values()),
            "byStatus": {
                status: counts.get(status, 0)
                for status in ("PENDING", "RUNNING", "COMPLETED", "FAILED")
            },
            "recentFailedJobs": [
                {
                    "jobType": row["job_type"],
                    "retryCount": int(row["retry_count"] or 0),
                    "updatedAt": _iso(row["updated_at"]),
                }
                for row in failures
            ],
        }
        return ChatDataResult(
            kind=spec.kind,
            data=data,
            data_as_of=_as_datetime(latest, self.now()),
            source_tables=("ai_report_jobs",),
            filters=spec.filters(),
        )

    async def _low_soh_batteries(self, spec: DataQuerySpec) -> ChatDataResult:
        threshold = spec.threshold if spec.threshold is not None else 70.0
        summary_sql = text(
            """
            SELECT
                COUNT(*) AS battery_count,
                AVG(soh_score) AS average_soh,
                MIN(soh_score) AS minimum_soh,
                MAX(last_inspected_at) AS latest_inspected_at
            FROM public."BATTERY_PASSPORT"
            WHERE soh_score IS NOT NULL AND soh_score < :threshold
            """
        )
        details_sql = text(
            """
            SELECT
                battery.car_id,
                car.nickname,
                car.model,
                battery.soh_score,
                battery.last_inspected_at
            FROM public."BATTERY_PASSPORT" battery
            JOIN public."CAR" car ON car.car_id = battery.car_id
            WHERE battery.soh_score IS NOT NULL AND battery.soh_score < :threshold
            ORDER BY battery.soh_score ASC, battery.car_id
            LIMIT :limit
            """
        )
        params = {"threshold": threshold, "limit": spec.limit or 10}
        async with self.sessions() as session:
            summary = (await session.execute(summary_sql, params)).mappings().one()
            detail_rows = (await session.execute(details_sql, params)).mappings().all()
        data = {
            "thresholdPercent": threshold,
            "batteryCount": int(summary["battery_count"] or 0),
            "averageSohPercent": _number(summary["average_soh"]),
            "minimumSohPercent": _number(summary["minimum_soh"]),
            "latestInspectionDate": _iso(summary["latest_inspected_at"]),
            "batteries": [
                {
                    "carId": str(row["car_id"]),
                    "vehicleName": row["nickname"] or row["model"],
                    "model": row["model"],
                    "sohPercent": _number(row["soh_score"]),
                    "lastInspectedAt": _iso(row["last_inspected_at"]),
                }
                for row in detail_rows
            ],
        }
        return ChatDataResult(
            kind=spec.kind,
            data=data,
            data_as_of=self.now(),
            source_tables=("BATTERY_PASSPORT", "CAR"),
            filters=spec.filters(),
        )

    async def _charging_summary(self, spec: DataQuerySpec) -> ChatDataResult:
        if spec.start_at is None or spec.end_at is None:
            raise ValueError("charging summary requires a period")
        statement = text(
            """
            SELECT
                COUNT(*) AS total_sessions,
                COUNT(*) FILTER (WHERE change_state = '충전완료') AS completed_sessions,
                COUNT(*) FILTER (
                    WHERE start_time IS NOT NULL
                      AND end_time IS NOT NULL
                      AND end_time >= start_time
                ) AS sessions_with_duration,
                COALESCE(SUM(
                    EXTRACT(EPOCH FROM (end_time - start_time)) / 3600.0
                ) FILTER (
                    WHERE start_time IS NOT NULL
                      AND end_time IS NOT NULL
                      AND end_time >= start_time
                ), 0) AS total_charging_hours,
                AVG(end_soc - start_soc) FILTER (
                    WHERE start_soc IS NOT NULL AND end_soc IS NOT NULL
                ) AS average_soc_change,
                COUNT(DISTINCT car_id) AS vehicle_count,
                MAX(COALESCE(end_time, start_time)) AS latest_session_at
            FROM public."CHARGING_SESSION"
            WHERE start_time >= :start_at AND start_time < :end_at
            """
        )
        params = {"start_at": spec.start_at, "end_at": spec.end_at}
        async with self.sessions() as session:
            row = (await session.execute(statement, params)).mappings().one()
        hours = _number(row["total_charging_hours"]) or 0.0
        average_soc = _number(row["average_soc_change"])
        data = {
            "totalSessions": int(row["total_sessions"] or 0),
            "completedSessions": int(row["completed_sessions"] or 0),
            "sessionsWithDuration": int(row["sessions_with_duration"] or 0),
            "totalChargingHours": round(hours, 1),
            "averageSocChangePercentPoints": (
                round(average_soc, 2) if average_soc is not None else None
            ),
            "vehicleCount": int(row["vehicle_count"] or 0),
        }
        return ChatDataResult(
            kind=spec.kind,
            data=data,
            data_as_of=_as_datetime(row["latest_session_at"], self.now()),
            source_tables=("CHARGING_SESSION",),
            filters=spec.filters(),
        )
