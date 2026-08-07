from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any
from uuid import UUID, NAMESPACE_URL, uuid4, uuid5

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .schemas import GeneratedReport, ReportType


@dataclass(frozen=True)
class ReportJob:
    job_id: UUID
    job_key: str
    job_type: ReportType
    car_id: UUID | None
    anomaly_id: UUID | None
    target_month: date | None
    retry_count: int


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class PostgresReportRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    async def enqueue_anomaly(self, anomaly_id: str | UUID) -> UUID:
        anomaly_uuid = UUID(str(anomaly_id))
        job_id = uuid4()
        async with self.sessions.begin() as session:
            car_id = await session.scalar(
                text(
                    'SELECT car_id FROM public."ANOMALY_LOGS" WHERE anomaly_id = :anomaly_id'
                ),
                {"anomaly_id": anomaly_uuid},
            )
            if car_id is None:
                raise LookupError("anomaly does not exist or has no vehicle")
            existing_or_created = await session.scalar(
                text(
                    """
                    INSERT INTO ai_report_jobs (
                        job_id, job_key, job_type, car_id, anomaly_id,
                        status, available_at
                    ) VALUES (
                        :job_id, :job_key, 'ANOMALY', :car_id, :anomaly_id,
                        'PENDING', NOW()
                    )
                    ON CONFLICT (job_key) DO UPDATE SET updated_at = NOW()
                    RETURNING job_id
                    """
                ),
                {
                    "job_id": job_id,
                    "job_key": f"ANOMALY:{anomaly_uuid}",
                    "car_id": car_id,
                    "anomaly_id": anomaly_uuid,
                },
            )
        return UUID(str(existing_or_created))

    async def enqueue_monthly(self, target_month: date) -> UUID:
        if target_month.day != 1:
            target_month = target_month.replace(day=1)
        job_id = uuid4()
        async with self.sessions.begin() as session:
            existing_or_created = await session.scalar(
                text(
                    """
                    INSERT INTO ai_report_jobs (
                        job_id, job_key, job_type, target_month,
                        status, available_at
                    ) VALUES (
                        :job_id, :job_key, 'MONTHLY', :target_month,
                        'PENDING', NOW()
                    )
                    ON CONFLICT (job_key) DO UPDATE SET updated_at = NOW()
                    RETURNING job_id
                    """
                ),
                {
                    "job_id": job_id,
                    "job_key": f"MONTHLY:GLOBAL:{target_month:%Y-%m}",
                    "target_month": target_month,
                },
            )
        return UUID(str(existing_or_created))

    async def enqueue_monthly_global(self, target_month: date) -> UUID:
        return await self.enqueue_monthly(target_month)

    async def claim_next(self) -> ReportJob | None:
        async with self.sessions.begin() as session:
            row = (
                await session.execute(
                    text(
                        """
                        WITH next_job AS (
                            SELECT job_id
                            FROM ai_report_jobs
                            WHERE status = 'PENDING'
                              AND available_at <= NOW()
                            ORDER BY created_at, job_id
                            FOR UPDATE SKIP LOCKED
                            LIMIT 1
                        )
                        UPDATE ai_report_jobs jobs
                        SET status = 'RUNNING', started_at = NOW(), updated_at = NOW()
                        FROM next_job
                        WHERE jobs.job_id = next_job.job_id
                        RETURNING jobs.job_id, jobs.job_key, jobs.job_type,
                                  jobs.car_id, jobs.anomaly_id, jobs.target_month,
                                  jobs.retry_count
                        """
                    )
                )
            ).mappings().first()
        if row is None:
            return None
        return ReportJob(
            job_id=UUID(str(row["job_id"])),
            job_key=row["job_key"],
            job_type=ReportType(row["job_type"]),
            car_id=(UUID(str(row["car_id"])) if row["car_id"] else None),
            anomaly_id=(
                UUID(str(row["anomaly_id"])) if row["anomaly_id"] else None
            ),
            target_month=row["target_month"],
            retry_count=int(row["retry_count"]),
        )

    async def requeue_stale_running(self, *, stale_seconds: int = 900) -> int:
        if stale_seconds <= 0:
            raise ValueError("stale_seconds must be positive")
        async with self.sessions.begin() as session:
            result = await session.execute(
                text(
                    """
                    UPDATE ai_report_jobs
                    SET status = 'PENDING',
                        available_at = NOW(),
                        error_message = 'worker lease expired; job requeued',
                        updated_at = NOW()
                    WHERE status = 'RUNNING'
                      AND started_at < NOW() - (:stale_seconds * INTERVAL '1 second')
                    """
                ),
                {"stale_seconds": stale_seconds},
            )
        return int(result.rowcount or 0)

    async def load_anomaly_facts(self, job: ReportJob) -> dict[str, Any]:
        if job.anomaly_id is None:
            raise ValueError("anomaly report job has no anomaly_id")
        async with self.sessions() as session:
            row = (
                await session.execute(
                    text(
                        """
                        SELECT
                            a.anomaly_id, a.abnormal_type, a.source_type,
                            a.trigger_value, a.detected_at, a.risk_level,
                            a.car_id, a.session_id,
                            c.model AS car_model, c.nickname AS car_nickname,
                            frame.observed_at AS frame_observed_at,
                            frame.hotspot_cell_index,
                            frame.hotspot_connector_index,
                            frame.ml_risk_level,
                            frame.physics_risk_level,
                            frame.final_risk_level,
                            frame.image_risk_level,
                            frame.image_confidence,
                            frame.raw_metrics,
                            frame.model_input,
                            COALESCE(soh_current.soh_score, battery.soh_score)
                                AS soh_score,
                            soh_previous.soh_score AS previous_soh_score,
                            battery.charge_cycles,
                            battery.current_temp AS passport_current_temp
                        FROM public."ANOMALY_LOGS" a
                        LEFT JOIN public."CAR" c ON c.car_id = a.car_id
                        LEFT JOIN public."BATTERY_PASSPORT" battery
                            ON battery.car_id = a.car_id
                        LEFT JOIN LATERAL (
                            SELECT history.soh_score
                            FROM public."BATTERY_SOH_HISTORY" history
                            WHERE history.battery_id = battery.battery_id
                              AND history.recorded_at <= a.detected_at
                            ORDER BY history.recorded_at DESC
                            LIMIT 1
                        ) soh_current ON TRUE
                        LEFT JOIN LATERAL (
                            SELECT history.soh_score
                            FROM public."BATTERY_SOH_HISTORY" history
                            WHERE history.battery_id = battery.battery_id
                              AND history.recorded_at <= a.detected_at
                            ORDER BY history.recorded_at DESC
                            OFFSET 1
                            LIMIT 1
                        ) soh_previous ON TRUE
                        LEFT JOIN LATERAL (
                            SELECT *
                            FROM public."TWIN_FRAMES" tf
                            WHERE tf.anomaly_id = a.anomaly_id
                            ORDER BY tf.observed_at DESC
                            LIMIT 1
                        ) frame ON TRUE
                        WHERE a.anomaly_id = :anomaly_id
                        """
                    ),
                    {"anomaly_id": job.anomaly_id},
                )
            ).mappings().first()
            temperature_rows = []
            if row is not None:
                temperature_rows = (
                    await session.execute(
                        text(
                            """
                            SELECT
                                date_trunc('hour', observed_at) AS observed_hour,
                                MAX((model_input->>'temp_max_c')::double precision)
                                    AS temperature_c
                            FROM public."TWIN_FRAMES"
                            WHERE car_id = :car_id
                              AND observed_at >=
                                  CAST(:detected_at AS timestamptz) - INTERVAL '24 hours'
                              AND observed_at <= CAST(:detected_at AS timestamptz)
                              AND model_input->>'temp_max_c'
                                  ~ '^-?[0-9]+(\\.[0-9]+)?$'
                            GROUP BY date_trunc('hour', observed_at)
                            ORDER BY observed_hour
                            """
                        ),
                        {
                            "car_id": row["car_id"],
                            "detected_at": row["detected_at"],
                        },
                    )
                ).mappings().all()
        if row is None:
            raise LookupError("anomaly data no longer exists")
        result = dict(row)
        result["model_input"] = _json_object(result.get("model_input"))
        result["raw_metrics"] = _json_object(result.get("raw_metrics"))
        result["temperature_history"] = [
            {
                "observed_at": item["observed_hour"],
                "temperature_c": item["temperature_c"],
            }
            for item in temperature_rows
        ]
        return result

    async def load_monthly_facts(self, job: ReportJob) -> dict[str, Any]:
        if job.target_month is None:
            raise ValueError("monthly report job has no target_month")
        period_start = job.target_month
        if period_start.month == 12:
            period_end = date(period_start.year + 1, 1, 1)
        else:
            period_end = date(period_start.year, period_start.month + 1, 1)

        async with self.sessions() as session:
            fleet = (
                await session.execute(
                    text(
                        """
                        SELECT COUNT(*)::integer AS vehicle_count
                        FROM public."CAR"
                        """
                    )
                )
            ).mappings().one()

            sessions = (
                await session.execute(
                    text(
                        """
                        SELECT
                            COUNT(*)::integer AS session_count,
                            COUNT(*) FILTER (WHERE end_time IS NOT NULL)::integer
                                AS completed_session_count,
                            COALESCE(SUM(
                                EXTRACT(EPOCH FROM (end_time - start_time)) / 60.0
                            ) FILTER (WHERE end_time IS NOT NULL), 0)::double precision
                                AS total_duration_minutes,
                            AVG((end_soc - start_soc)::double precision)
                                FILTER (WHERE start_soc IS NOT NULL AND end_soc IS NOT NULL)
                                AS average_soc_change
                        FROM public."CHARGING_SESSION"
                        WHERE start_time >= :period_start
                          AND start_time < :period_end
                        """
                    ),
                    {
                        "period_start": period_start,
                        "period_end": period_end,
                    },
                )
            ).mappings().one()
            anomaly_rows = (
                await session.execute(
                    text(
                        """
                        SELECT risk_level, COUNT(*)::integer AS count
                        FROM public."ANOMALY_LOGS"
                        WHERE detected_at >= :period_start
                          AND detected_at < :period_end
                        GROUP BY risk_level
                        ORDER BY risk_level
                        """
                    ),
                    {
                        "period_start": period_start,
                        "period_end": period_end,
                    },
                )
            ).mappings().all()
            sensors = (
                await session.execute(
                    text(
                        """
                        SELECT
                            COUNT(*)::integer AS sample_count,
                            AVG(CASE
                                WHEN model_input->>'temp_max_c' ~ '^-?[0-9]+(\\.[0-9]+)?$'
                                THEN (model_input->>'temp_max_c')::double precision
                            END) AS average_max_temperature_c,
                            MAX(CASE
                                WHEN model_input->>'temp_max_c' ~ '^-?[0-9]+(\\.[0-9]+)?$'
                                THEN (model_input->>'temp_max_c')::double precision
                            END) AS highest_temperature_c
                        FROM public."TWIN_FRAMES"
                        WHERE observed_at >= :period_start
                          AND observed_at < :period_end
                        """
                    ),
                    {
                        "period_start": period_start,
                        "period_end": period_end,
                    },
                )
            ).mappings().one()

        return {
            "fleet": dict(fleet),
            "periodStart": period_start,
            "periodEndExclusive": period_end,
            "chargingSessions": dict(sessions),
            "anomalies": [dict(row) for row in anomaly_rows],
            "sensorSummary": dict(sensors),
        }

    async def save_report(
        self,
        job: ReportJob,
        title: str,
        report: GeneratedReport,
    ) -> UUID:
        report_id = uuid5(NAMESPACE_URL, f"ev-ai-report:{job.job_key}")
        report_json = report.model_dump_json(by_alias=True)
        async with self.sessions.begin() as session:
            await session.execute(
                text(
                    """
                    INSERT INTO public."AI_REPORTS" (
                        report_id, title, report_data, report_type, created_at,
                        car_id, anomaly_id, is_read
                    ) VALUES (
                        :report_id, :title, CAST(:report_data AS jsonb), :report_type,
                        NOW(), :car_id, :anomaly_id, FALSE
                    )
                    ON CONFLICT (report_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        report_data = EXCLUDED.report_data,
                        report_type = EXCLUDED.report_type,
                        anomaly_id = EXCLUDED.anomaly_id
                    """
                ),
                {
                    "report_id": report_id,
                    "title": title,
                    "report_data": report_json,
                    "report_type": report.report_type.public_value,
                    "car_id": job.car_id,
                    "anomaly_id": job.anomaly_id,
                },
            )
        return report_id

    async def mark_completed(self, job_id: UUID, report_id: UUID) -> None:
        async with self.sessions.begin() as session:
            await session.execute(
                text(
                    """
                    UPDATE ai_report_jobs
                    SET status = 'COMPLETED', report_id = :report_id,
                        completed_at = NOW(), error_message = NULL, updated_at = NOW()
                    WHERE job_id = :job_id
                    """
                ),
                {"job_id": job_id, "report_id": report_id},
            )

    async def mark_failed(
        self,
        job: ReportJob,
        error: Exception,
        *,
        max_retries: int,
    ) -> None:
        next_retry = job.retry_count + 1
        terminal = next_retry >= max_retries
        async with self.sessions.begin() as session:
            await session.execute(
                text(
                    """
                    UPDATE ai_report_jobs
                    SET status = :status,
                        retry_count = :retry_count,
                        error_message = :error_message,
                        available_at = CASE
                            WHEN :terminal THEN available_at
                            ELSE NOW() + (:delay_seconds * INTERVAL '1 second')
                        END,
                        completed_at = CASE WHEN :terminal THEN NOW() ELSE NULL END,
                        updated_at = NOW()
                    WHERE job_id = :job_id
                    """
                ),
                {
                    "job_id": job.job_id,
                    "status": "FAILED" if terminal else "PENDING",
                    "retry_count": next_retry,
                    "error_message": str(error)[:2_000],
                    "terminal": terminal,
                    "delay_seconds": min(300, 2**next_retry * 5),
                },
            )
