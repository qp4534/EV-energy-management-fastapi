import asyncio
from datetime import date, datetime, timezone
from uuid import UUID

import pytest

from app.ai.config import AISettings
from app.ai.contracts import RetrievedChunk
from app.reporting.repository import ReportJob
from app.reporting.schemas import ReportType
from app.reporting.service import ReportGenerationService
from app.reporting.worker import previous_month, run_loop


NOW = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)


def test_previous_month_handles_year_boundary() -> None:
    assert previous_month(datetime(2026, 1, 15, tzinfo=timezone.utc)) == date(
        2025, 12, 1
    )


class FakeReportData:
    def __init__(self, anomaly=None, monthly=None):
        self.anomaly = anomaly or {}
        self.monthly = monthly or {}

    async def load_anomaly_facts(self, job):
        return self.anomaly

    async def load_monthly_facts(self, job):
        return self.monthly


class FakeRag:
    def __init__(self, chunks=None):
        self.chunks = list(chunks or [])

    async def search(self, query, *, route, top_k=None):
        assert route == "REPORT"
        return self.chunks


class FakeGenerator:
    def __init__(self, response=None, *, fail=False):
        self.response = response or (
            '{"summary":"AI 요약","interpretation":"확정 원인은 알 수 없습니다.",'
            '"recommendedActions":["안전 지침을 확인하세요."]}'
        )
        self.fail = fail

    async def generate(self, system_prompt, user_prompt, *, purpose, json_mode=False):
        assert purpose == "report"
        assert json_mode is True
        if self.fail:
            raise RuntimeError("model down")
        return self.response


def evidence() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="safety-1",
        document_id="safety",
        source_title="안전 가이드",
        source_type="technical_guide",
        content="안전 지침",
        score=0.9,
        page=3,
    )


def job(report_type: ReportType) -> ReportJob:
    return ReportJob(
        job_id=UUID("11111111-1111-1111-1111-111111111111"),
        job_key=f"{report_type}:test",
        job_type=report_type,
        car_id=UUID("22222222-2222-2222-2222-222222222222"),
        anomaly_id=(
            UUID("33333333-3333-3333-3333-333333333333")
            if report_type == ReportType.ANOMALY
            else None
        ),
        target_month=date(2026, 7, 1) if report_type == ReportType.MONTHLY else None,
        retry_count=0,
    )


@pytest.mark.asyncio
async def test_anomaly_report_keeps_db_metrics_and_has_no_human_signature_fields() -> None:
    facts = {
        "detected_at": NOW,
        "abnormal_type": "BMS_SAFETY_ALERT",
        "risk_level": "경고",
        "frame_observed_at": NOW,
        "model_input": {"temp_max_c": 61.2, "voltage_v": 3.91},
    }
    service = ReportGenerationService(
        FakeReportData(anomaly=facts),
        FakeRag([evidence()]),
        FakeGenerator(),
        now=lambda: NOW,
    )

    _, report = await service.generate(job(ReportType.ANOMALY))
    payload = report.model_dump(mode="json", by_alias=True)

    assert payload["reportType"] == "이상"
    assert report.risk_level == "WARNING"
    assert report.llm_enhanced is True
    metric_items = payload["sections"][1]["items"]
    assert {item["label"]: item["value"] for item in metric_items}["최고 배터리 온도"] == 61.2
    serialized = report.model_dump_json(by_alias=True)
    assert "작성자" not in serialized
    assert "검토자" not in serialized
    assert "서명" not in serialized
    assert payload["actions"] == []


@pytest.mark.asyncio
async def test_monthly_report_uses_deterministic_aggregates_when_llm_is_unavailable() -> None:
    facts = {
        "periodStart": date(2026, 7, 1),
        "periodEndExclusive": date(2026, 8, 1),
        "chargingSessions": {
            "session_count": 4,
            "completed_session_count": 3,
            "total_duration_minutes": 185.5,
            "average_soc_change": 23.25,
        },
        "anomalies": [{"risk_level": "주의", "count": 2}],
        "sensorSummary": {
            "sample_count": 2,
            "highest_temperature_c": 44.2,
            "average_max_temperature_c": 40.1,
        },
    }
    service = ReportGenerationService(
        FakeReportData(monthly=facts),
        FakeRag([evidence()]),
        FakeGenerator(fail=True),
        now=lambda: NOW,
    )

    _, report = await service.generate(job(ReportType.MONTHLY))

    assert report.model_dump(mode="json", by_alias=True)["reportType"] == "월간보고서"
    assert report.llm_enhanced is False
    assert report.risk_level == "CAUTION"
    metrics = report.sections[1].items
    assert {item["label"]: item["value"] for item in metrics}["충전 세션"] == 4
    assert report.period.from_date == date(2026, 7, 1)
    assert report.period.to_date == date(2026, 7, 31)


@pytest.mark.asyncio
async def test_report_without_rag_does_not_call_freeform_llm() -> None:
    class MustNotRun:
        async def generate(self, *args, **kwargs):
            raise AssertionError("LLM must not run without RAG evidence")

    facts = {
        "periodStart": date(2026, 7, 1),
        "periodEndExclusive": date(2026, 8, 1),
        "chargingSessions": {},
        "anomalies": [],
        "sensorSummary": {},
    }
    service = ReportGenerationService(
        FakeReportData(monthly=facts), FakeRag(), MustNotRun(), now=lambda: NOW
    )

    _, report = await service.generate(job(ReportType.MONTHLY))

    assert report.llm_enhanced is False
    assert "ragEvidence" in report.missing_fields


@pytest.mark.asyncio
async def test_embedded_report_worker_stops_without_waiting_for_poll_timeout() -> None:
    stop_event = asyncio.Event()

    class EmptyQueue:
        async def requeue_stale_running(self):
            return 0

        async def enqueue_monthly_for_all(self, target_month):
            stop_event.set()
            return 0

        async def claim_next(self):
            return None

    settings = AISettings(
        database_url="postgresql+asyncpg://test",
        report_worker_poll_seconds=60.0,
    )

    await run_loop(
        EmptyQueue(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        settings,
        stop_event,
    )

    assert stop_event.is_set()
