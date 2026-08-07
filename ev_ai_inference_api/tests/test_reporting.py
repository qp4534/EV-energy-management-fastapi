import asyncio
from datetime import date, datetime, timedelta, timezone
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
        car_id=(
            UUID("22222222-2222-2222-2222-222222222222")
            if report_type == ReportType.ANOMALY
            else None
        ),
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
        "raw_metrics": {
            "twin_frame": {"voltage_mv": [3800, 3910, 4140]}
        },
        "soh_score": 94.2,
        "previous_soh_score": 96.0,
        "charge_cycles": 342,
        "temperature_history": [
            {"observed_at": NOW - timedelta(hours=1), "temperature_c": 38.4},
            {"observed_at": NOW, "temperature_c": 61.2},
        ],
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
    detection_section = next(
        section for section in payload["sections"] if section["title"] == "이상 감지 정보"
    )
    detection_by_label = {
        item["label"]: item["value"] for item in detection_section["items"]
    }
    assert detection_by_label["발생 시각"] == "2026-08-06 21:00:00 KST"
    assert detection_by_label["위험등급"] == "경고"
    assert detection_by_label["이상 유형"] == "BMS_SAFETY_ALERT"
    metric_section = next(
        section for section in payload["sections"] if section["title"] == "이상 지표"
    )
    metric_items = metric_section["items"]
    metrics_by_label = {item["label"]: item for item in metric_items}
    assert metrics_by_label["배터리 온도"]["value"] == 61.2
    assert metrics_by_label["SOH 변동"]["value"] == "-1.8"
    assert metrics_by_label["전압 편차"]["value"] == 0.34
    assert metrics_by_label["누적 충전 사이클"]["value"] == 342
    chart = next(section for section in payload["sections"] if section["type"] == "lineChart")
    assert chart["title"] == "최근 24시간 온도 추이"
    assert chart["datasets"][0]["data"] == [38.4, 61.2]
    assert chart["unit"] == "°C"
    assert any(section["title"] == "원인 분석" for section in payload["sections"])
    assert any(section["title"] == "권장 조치" for section in payload["sections"])
    serialized = report.model_dump_json(by_alias=True)
    assert "작성자" not in serialized
    assert "검토자" not in serialized
    assert "서명" not in serialized
    assert payload["actions"] == []


@pytest.mark.asyncio
async def test_anomaly_report_uses_voltage_trigger_when_cell_values_are_missing() -> None:
    facts = {
        "detected_at": NOW,
        "abnormal_type": "셀 전압 불균형",
        "source_type": "BMS",
        "trigger_value": "전압편차 0.164V",
        "risk_level": "경고",
        "raw_metrics": {},
    }
    service = ReportGenerationService(
        FakeReportData(anomaly=facts),
        FakeRag(),
        FakeGenerator(),
        now=lambda: NOW,
    )

    _, report = await service.generate(job(ReportType.ANOMALY))
    payload = report.model_dump(mode="json", by_alias=True)
    metric_section = next(
        section for section in payload["sections"] if section["title"] == "이상 지표"
    )
    voltage_metric = next(
        item for item in metric_section["items"] if item["label"] == "전압 편차"
    )

    assert voltage_metric["value"] == 0.164
    assert voltage_metric["unit"] == "V"
    assert voltage_metric["caption"] == "이상 감지 로그 기준"
    assert "cellVoltages" in payload["missingFields"]


@pytest.mark.asyncio
async def test_anomaly_report_does_not_treat_unrelated_trigger_as_voltage() -> None:
    facts = {
        "detected_at": NOW,
        "abnormal_type": "배터리 과열 징후",
        "source_type": "THERMAL",
        "trigger_value": "최고온도 51.3°C",
        "risk_level": "경고",
        "raw_metrics": {},
    }
    service = ReportGenerationService(
        FakeReportData(anomaly=facts),
        FakeRag(),
        FakeGenerator(),
        now=lambda: NOW,
    )

    _, report = await service.generate(job(ReportType.ANOMALY))
    payload = report.model_dump(mode="json", by_alias=True)
    metric_section = next(
        section for section in payload["sections"] if section["title"] == "이상 지표"
    )
    voltage_metric = next(
        item for item in metric_section["items"] if item["label"] == "전압 편차"
    )

    assert voltage_metric["value"] == "확인 불가"
    assert voltage_metric["caption"] == "셀별 전압 데이터 없음"


@pytest.mark.asyncio
async def test_monthly_report_uses_deterministic_aggregates_when_llm_is_unavailable() -> None:
    facts = {
        "periodStart": date(2026, 7, 1),
        "periodEndExclusive": date(2026, 8, 1),
        "fleet": {"vehicle_count": 140},
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
    metrics_by_label = {item["label"]: item["value"] for item in metrics}
    assert metrics_by_label["전체 차량"] == 140
    assert metrics_by_label["충전 세션"] == 4
    assert metrics_by_label["총 충전 시간"] == 3.1
    duration_metric = next(item for item in metrics if item["label"] == "총 충전 시간")
    assert duration_metric["unit"] == "시간"
    assert report.period.from_date == date(2026, 7, 1)
    assert report.period.to_date == date(2026, 7, 31)


@pytest.mark.asyncio
async def test_monthly_report_sends_charge_duration_to_llm_in_hours() -> None:
    class CapturingGenerator:
        user_prompt = ""

        async def generate(
            self, system_prompt, user_prompt, *, purpose, json_mode=False
        ):
            self.user_prompt = user_prompt
            return (
                '{"summary":"월간 요약","interpretation":"월간 해석",'
                '"recommendedActions":[]}'
            )

    facts = {
        "periodStart": date(2026, 7, 1),
        "periodEndExclusive": date(2026, 8, 1),
        "fleet": {"vehicle_count": 140},
        "chargingSessions": {
            "session_count": 800,
            "completed_session_count": 670,
            "total_duration_minutes": 174764.8,
            "average_soc_change": 49.0,
        },
        "anomalies": [],
        "sensorSummary": {},
    }
    generator = CapturingGenerator()
    service = ReportGenerationService(
        FakeReportData(monthly=facts),
        FakeRag([evidence()]),
        generator,
        now=lambda: NOW,
    )

    _, report = await service.generate(job(ReportType.MONTHLY))
    payload = report.model_dump(mode="json", by_alias=True)
    metric_section = next(
        section for section in payload["sections"] if section["title"] == "주요 지표"
    )
    duration_metric = next(
        item for item in metric_section["items"] if item["label"] == "총 충전 시간"
    )

    assert duration_metric["value"] == 2912.7
    assert duration_metric["unit"] == "시간"
    assert '"total_duration_hours": 2912.7' in generator.user_prompt
    assert "total_duration_minutes" not in generator.user_prompt


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

        async def enqueue_monthly_global(self, target_month):
            stop_event.set()
            return UUID("55555555-5555-5555-5555-555555555555")

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
