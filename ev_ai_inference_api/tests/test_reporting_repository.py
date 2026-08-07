from datetime import datetime, timezone
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from app.reporting.repository import PostgresReportRepository, ReportJob
from app.reporting.schemas import GeneratedReport, ReportType


class FakeSession:
    def __init__(self, existing_report_id: UUID | None) -> None:
        self.existing_report_id = existing_report_id
        self.scalar_calls: list[tuple[str, dict]] = []
        self.execute_calls: list[tuple[str, dict]] = []

    async def scalar(self, statement, params):
        self.scalar_calls.append((str(statement), params))
        return self.existing_report_id

    async def execute(self, statement, params):
        self.execute_calls.append((str(statement), params))


class FakeTransaction:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> FakeSession:
        return self.session

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class FakeSessions:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self.session)


def anomaly_job() -> ReportJob:
    return ReportJob(
        job_id=UUID("11111111-1111-1111-1111-111111111111"),
        job_key="ANOMALY:33333333-3333-3333-3333-333333333333",
        job_type=ReportType.ANOMALY,
        car_id=UUID("22222222-2222-2222-2222-222222222222"),
        anomaly_id=UUID("33333333-3333-3333-3333-333333333333"),
        target_month=None,
        retry_count=0,
    )


def generated_report() -> GeneratedReport:
    return GeneratedReport(
        report_type=ReportType.ANOMALY,
        data_as_of=datetime(2026, 8, 7, tzinfo=timezone.utc),
        risk_level="WARNING",
        sections=[],
    )


@pytest.mark.asyncio
async def test_save_anomaly_report_reuses_existing_report_id() -> None:
    existing_report_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    session = FakeSession(existing_report_id)
    repository = PostgresReportRepository(FakeSessions(session))

    saved_report_id = await repository.save_report(
        anomaly_job(),
        "기존 이상 보고서 갱신",
        generated_report(),
    )

    assert saved_report_id == existing_report_id
    assert len(session.scalar_calls) == 1
    select_sql, select_params = session.scalar_calls[0]
    assert 'FROM public."AI_REPORTS"' in select_sql
    assert select_params["anomaly_id"] == anomaly_job().anomaly_id
    assert session.execute_calls[0][1]["report_id"] == existing_report_id


@pytest.mark.asyncio
async def test_save_new_anomaly_report_uses_deterministic_report_id() -> None:
    session = FakeSession(None)
    repository = PostgresReportRepository(FakeSessions(session))
    job = anomaly_job()

    saved_report_id = await repository.save_report(
        job,
        "신규 이상 보고서",
        generated_report(),
    )

    expected = uuid5(NAMESPACE_URL, f"ev-ai-report:{job.job_key}")
    assert saved_report_id == expected
    assert session.execute_calls[0][1]["report_id"] == expected
