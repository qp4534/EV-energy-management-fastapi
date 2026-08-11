from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

import pytest

from app.chatbot.data_queries import DataQueryKind, DataQuerySpec
from app.chatbot.data_repository import PostgresChatDataRepository


NOW = datetime(2026, 8, 11, 6, 0, tzinfo=timezone.utc)
START = datetime(2026, 7, 1, tzinfo=timezone.utc)
END = datetime(2026, 8, 1, tzinfo=timezone.utc)


class FakeResult:
    def __init__(self, *, one=None, all=None):
        self.one_value = one
        self.all_value = list(all or [])

    def mappings(self):
        return self

    def one(self):
        return self.one_value

    def all(self):
        return self.all_value


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def execute(self, statement, params):
        self.calls.append((str(statement), params))
        return self.responses.pop(0)


class FakeSessions:
    def __init__(self, responses):
        self.session = FakeSession(responses)

    def __call__(self):
        return self.session


def repository(*responses):
    sessions = FakeSessions(responses)
    return PostgresChatDataRepository(sessions, now=lambda: NOW), sessions


@pytest.mark.asyncio
async def test_risk_overview_maps_latest_twin_counts() -> None:
    repo, sessions = repository(
        FakeResult(
            one={
                "total_vehicles": 5,
                "vehicles_with_twin": 4,
                "fresh_vehicles": 2,
                "normal_count": 1,
                "caution_count": 1,
                "warning_count": 1,
                "emergency_count": 1,
                "stale_count": 2,
                "latest_observed_at": NOW,
            }
        )
    )

    result = await repo.fetch(DataQuerySpec(kind=DataQueryKind.RISK_OVERVIEW))

    assert result.data["totalVehicles"] == 5
    assert result.data["unknown"] == 3
    assert result.data["staleVehicles"] == 2
    assert result.source_tables == ("CAR", "TWIN_FRAMES")
    assert 'public."TWIN_FRAMES"' in sessions.session.calls[0][0]


@pytest.mark.asyncio
async def test_anomaly_summary_preserves_db_counts_and_types() -> None:
    repo, _ = repository(
        FakeResult(
            one={
                "total_count": 9,
                "normal_count": 1,
                "caution_count": 4,
                "warning_count": 3,
                "emergency_count": 1,
                "affected_vehicles": 3,
                "latest_detected_at": NOW,
            }
        ),
        FakeResult(
            all=[
                {"abnormal_type": "셀 전압 불균형", "event_count": 6},
                {"abnormal_type": "온도 상승", "event_count": 3},
            ]
        ),
    )
    spec = DataQuerySpec(
        kind=DataQueryKind.ANOMALY_SUMMARY,
        period_label="2026-07",
        start_at=START,
        end_at=END,
    )

    result = await repo.fetch(spec)

    assert result.data["totalEvents"] == 9
    assert result.data["byRiskLevel"]["경고"] == 3
    assert result.data["byType"][0] == {"type": "셀 전압 불균형", "count": 6}


@pytest.mark.asyncio
async def test_report_job_status_does_not_expose_error_messages() -> None:
    repo, _ = repository(
        FakeResult(
            all=[
                {"status": "COMPLETED", "job_count": 8, "latest_updated_at": NOW},
                {"status": "FAILED", "job_count": 2, "latest_updated_at": NOW},
            ]
        ),
        FakeResult(
            all=[
                {"job_type": "ANOMALY", "retry_count": 3, "updated_at": NOW},
            ]
        ),
    )
    spec = DataQuerySpec(
        kind=DataQueryKind.REPORT_JOB_STATUS,
        start_at=START,
        end_at=END,
    )

    result = await repo.fetch(spec)

    assert result.data["totalJobs"] == 10
    assert result.data["byStatus"]["FAILED"] == 2
    assert "errorMessage" not in result.data["recentFailedJobs"][0]


@pytest.mark.asyncio
async def test_low_soh_query_returns_limited_vehicle_details() -> None:
    car_id = UUID("11111111-1111-1111-1111-111111111111")
    repo, _ = repository(
        FakeResult(
            one={
                "battery_count": 1,
                "average_soh": Decimal("64.5"),
                "minimum_soh": Decimal("64.5"),
                "latest_inspected_at": None,
            }
        ),
        FakeResult(
            all=[
                {
                    "car_id": car_id,
                    "nickname": "테스트 차량",
                    "model": "EV6",
                    "soh_score": Decimal("64.5"),
                    "last_inspected_at": None,
                }
            ]
        ),
    )

    result = await repo.fetch(
        DataQuerySpec(
            kind=DataQueryKind.LOW_SOH_BATTERIES,
            threshold=70.0,
            limit=10,
        )
    )

    assert result.data["batteryCount"] == 1
    assert result.data["batteries"][0]["sohPercent"] == 64.5
    assert result.data["batteries"][0]["vehicleName"] == "테스트 차량"


@pytest.mark.asyncio
async def test_charging_summary_reports_hours_instead_of_minutes() -> None:
    repo, _ = repository(
        FakeResult(
            one={
                "total_sessions": 800,
                "completed_sessions": 670,
                "sessions_with_duration": 670,
                "total_charging_hours": Decimal("2912.666"),
                "average_soc_change": Decimal("48.966"),
                "vehicle_count": 140,
                "latest_session_at": NOW,
            }
        )
    )
    spec = DataQuerySpec(
        kind=DataQueryKind.CHARGING_SUMMARY,
        period_label="2026-07",
        start_at=START,
        end_at=END,
    )

    result = await repo.fetch(spec)

    assert result.data["totalSessions"] == 800
    assert result.data["totalChargingHours"] == 2912.7
    assert result.data["averageSocChangePercentPoints"] == 48.97
