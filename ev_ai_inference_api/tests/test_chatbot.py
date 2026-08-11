import logging
from datetime import datetime, timedelta, timezone

import pytest

from app.ai.config import AISettings
from app.ai.contracts import RetrievedChunk
from app.chatbot.data_queries import (
    ChatDataResult,
    DataQueryKind,
    detect_data_query,
)
from app.chatbot.schemas import ChatMessageRequest
from app.chatbot.service import ChatbotService
from app.chatbot.supervisor import ChatRoute, ChatSupervisor


NOW = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)


class FakeRag:
    def __init__(self, chunks=None, *, fail=False):
        self.chunks = list(chunks or [])
        self.fail = fail
        self.calls = []

    async def search(self, query, *, route, top_k=None):
        self.calls.append((query, route, top_k))
        if self.fail:
            raise RuntimeError("database down")
        return self.chunks


class FakeGenerator:
    def __init__(self, response="생성된 답변", *, fail=False):
        self.response = response
        self.fail = fail
        self.calls = []

    async def generate(self, system_prompt, user_prompt, *, purpose, json_mode=False):
        self.calls.append((system_prompt, user_prompt, purpose, json_mode))
        if self.fail:
            raise RuntimeError("model down")
        return self.response


class FakeVehicleState:
    def __init__(self, value=None, *, fail=False):
        self.value = value
        self.fail = fail

    async def latest(self, vehicle_id):
        if self.fail:
            raise RuntimeError("inference down")
        return self.value


class FakeDataProvider:
    def __init__(self, value=None, *, fail=False):
        self.value = value
        self.fail = fail
        self.calls = []

    async def fetch(self, spec):
        self.calls.append(spec)
        if self.fail:
            raise RuntimeError("project database down")
        return self.value


def settings(**updates) -> AISettings:
    values = {"database_url": "postgresql+asyncpg://test"}
    values.update(updates)
    return AISettings(**values)


def chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="guide-1",
        document_id="guide",
        source_title="전기자동차 화재대응 가이드",
        source_type="technical_guide",
        content="연기나 불꽃이 있으면 안전거리를 확보하고 신고한다.",
        score=0.91,
        page=3,
    )


def service(
    rag,
    generator,
    vehicle=None,
    data=None,
    **setting_updates,
) -> ChatbotService:
    return ChatbotService(
        ChatSupervisor(),
        rag,
        generator,
        vehicle or FakeVehicleState(),
        settings(**setting_updates),
        data_provider=data,
        now=lambda: NOW,
    )


def test_supervisor_uses_safety_first_deterministic_routes() -> None:
    supervisor = ChatSupervisor()
    assert supervisor.classify("충전기에서 연기가 나요") == ChatRoute.EMERGENCY
    assert supervisor.classify("한국전기설비규정 조항 알려줘") == ChatRoute.LEGAL
    assert supervisor.classify("지금 내 차 배터리 상태는?") == ChatRoute.VEHICLE_STATUS
    assert supervisor.classify("안녕하세요") == ChatRoute.GENERAL
    assert supervisor.classify("충전기는 어떻게 사용해?") == ChatRoute.RAG


def test_project_data_parser_uses_explicit_month_and_soh_threshold() -> None:
    anomaly = detect_data_query("2026년 7월 이상 발생 건수를 알려줘", now=NOW)
    low_soh = detect_data_query("SOH 65% 미만 배터리가 몇 개인지 알려줘", now=NOW)

    assert anomaly is not None
    assert anomaly.kind == DataQueryKind.ANOMALY_SUMMARY
    assert anomaly.period_label == "2026-07"
    assert anomaly.start_at is not None
    assert anomaly.start_at.month == 7
    assert low_soh is not None
    assert low_soh.kind == DataQueryKind.LOW_SOH_BATTERIES
    assert low_soh.threshold == 65.0


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("현재 위험등급별 차량 수를 알려줘", DataQueryKind.RISK_OVERVIEW),
        ("이번 달 이상 발생 건수를 유형별로 정리해줘", DataQueryKind.ANOMALY_SUMMARY),
        ("최근 실패한 AI 보고서 작업이 있는지 알려줘", DataQueryKind.REPORT_JOB_STATUS),
        ("SOH가 70% 미만인 배터리가 몇 개인지 알려줘", DataQueryKind.LOW_SOH_BATTERIES),
        ("이번 달 전체 충전 세션과 총 충전 시간을 알려줘", DataQueryKind.CHARGING_SUMMARY),
    ],
)
def test_supported_project_demo_questions_are_detected(
    question: str,
    expected: DataQueryKind,
) -> None:
    spec = detect_data_query(question, now=NOW)

    assert spec is not None
    assert spec.kind == expected


def risk_data() -> ChatDataResult:
    return ChatDataResult(
        kind=DataQueryKind.RISK_OVERVIEW,
        data={
            "totalVehicles": 4,
            "vehiclesWithTwin": 3,
            "freshVehicles": 3,
            "normal": 1,
            "caution": 1,
            "warning": 1,
            "emergency": 0,
            "unknown": 1,
            "staleVehicles": 0,
            "staleAfterMinutes": 5,
        },
        data_as_of=NOW,
        source_tables=("CAR", "TWIN_FRAMES"),
        filters={},
    )


@pytest.mark.asyncio
async def test_admin_project_data_query_uses_allow_listed_provider() -> None:
    data = FakeDataProvider(risk_data())
    generator = FakeGenerator("현재 경고 차량은 1대입니다.")

    result = await service(FakeRag(), generator, data=data).answer(
        ChatMessageRequest(
            message="현재 위험등급별 차량 수를 알려줘",
            actor_role="관리자",
            user_id="admin-1",
        )
    )

    assert result.route == "ADMIN_DATA"
    assert result.safety_level == "WARNING"
    assert result.data_as_of == NOW
    assert result.metadata["dataQuery"]["sourceTables"] == ["CAR", "TWIN_FRAMES"]
    assert result.sources[0].source_type == "project_database"
    assert result.sources[0].title == "프로젝트 DB · 차량 정보"
    assert data.calls[0].kind == DataQueryKind.RISK_OVERVIEW
    assert '"totalVehicles": 4' in generator.calls[0][1]


@pytest.mark.asyncio
async def test_operator_project_data_query_uses_operator_route() -> None:
    data = FakeDataProvider(risk_data())

    result = await service(FakeRag(), FakeGenerator("관제 요약"), data=data).answer(
        ChatMessageRequest(
            message="위험 차량 현황을 전체로 알려줘",
            actor_role="관제자",
            user_id="operator-1",
        )
    )

    assert result.route == "OPERATOR_DATA"
    assert result.answer == "관제 요약"


@pytest.mark.asyncio
async def test_regular_user_cannot_query_fleet_data() -> None:
    data = FakeDataProvider(risk_data())

    result = await service(FakeRag(), FakeGenerator(), data=data).answer(
        ChatMessageRequest(
            message="현재 위험등급별 차량 수를 알려줘",
            actor_role="이용자",
        )
    )

    assert result.route == "DATA_QUERY"
    assert result.missing_fields == ["authorizedRole"]
    assert data.calls == []


@pytest.mark.asyncio
async def test_operator_role_without_authenticated_user_is_rejected() -> None:
    data = FakeDataProvider(risk_data())

    result = await service(FakeRag(), FakeGenerator(), data=data).answer(
        ChatMessageRequest(
            message="현재 위험등급별 차량 수를 알려줘",
            actor_role="관제자",
        )
    )

    assert result.route == "OPERATOR_DATA"
    assert result.missing_fields == ["authenticatedUser"]
    assert data.calls == []


@pytest.mark.asyncio
async def test_project_data_query_has_deterministic_fallback_when_llm_fails() -> None:
    result = await service(
        FakeRag(),
        FakeGenerator(fail=True),
        data=FakeDataProvider(risk_data()),
    ).answer(
        ChatMessageRequest(
            message="현재 위험등급별 차량 수를 알려줘",
            actor_role="관리자",
            user_id="admin-1",
        )
    )

    assert "전체 차량 4대" in result.answer
    assert result.fallback_used is True
    assert result.missing_fields == ["languageModel"]


@pytest.mark.asyncio
async def test_emergency_never_uses_general_fallback_when_rag_is_down() -> None:
    generator = FakeGenerator()
    result = await service(FakeRag(fail=True), generator).answer(
        ChatMessageRequest(message="차에서 연기가 나요")
    )

    assert result.route == "EMERGENCY"
    assert result.fallback_used is False
    assert "119" in result.answer
    assert result.missing_fields == ["rag"]
    assert generator.calls == []


@pytest.mark.asyncio
async def test_grounded_answer_returns_citation() -> None:
    result = await service(FakeRag([chunk()]), FakeGenerator("근거 기반 안내")).answer(
        ChatMessageRequest(message="충전 중 주의할 점은?")
    )

    assert result.route == "RAG"
    assert result.answer == "근거 기반 안내"
    assert result.sources[0].title == "전기자동차 화재대응 가이드"
    assert result.sources[0].page == 3
    assert result.fallback_used is False


@pytest.mark.asyncio
async def test_rag_miss_uses_general_fallback_only_for_noncritical_question() -> None:
    result = await service(FakeRag(), FakeGenerator("일반 안내")).answer(
        ChatMessageRequest(message="전기차란 뭐야?")
    )

    assert result.route == "GENERAL"
    assert result.answer == "일반 안내"
    assert result.fallback_used is True


@pytest.mark.asyncio
async def test_legal_question_without_evidence_does_not_fallback() -> None:
    generator = FakeGenerator()
    result = await service(FakeRag(), generator).answer(
        ChatMessageRequest(message="현재 법령상 신고 의무가 있어?")
    )

    assert result.route == "LEGAL"
    assert result.fallback_used is False
    assert "근거" in result.answer
    assert generator.calls == []


@pytest.mark.asyncio
async def test_vehicle_state_rejects_stale_snapshot() -> None:
    state = {
        "observed_at": (NOW - timedelta(minutes=2)).isoformat(),
        "final_risk_level": 0,
    }
    result = await service(
        FakeRag(), FakeGenerator(), FakeVehicleState(state)
    ).answer(
        ChatMessageRequest(
            message="지금 내 차 배터리 상태는?", vehicle_id="car-1"
        )
    )

    assert result.safety_level == "UNKNOWN"
    assert result.missing_fields == ["freshVehicleState"]
    assert "오래" in result.answer


@pytest.mark.asyncio
async def test_vehicle_state_uses_current_snapshot_without_inventing_metrics() -> None:
    state = {
        "observed_at": NOW.isoformat(),
        "sequence": 10,
        "final_risk_level": 1,
        "temperature_decic": [350, 420],
        "voltage_mv": [3800, 3900],
        "connector_temperature_decic": [400, 450, 410],
    }
    result = await service(
        FakeRag([chunk()]), FakeGenerator("현재 주의 단계입니다."), FakeVehicleState(state)
    ).answer(
        ChatMessageRequest(
            message="지금 내 차 배터리 상태는?", vehicle_id="car-1"
        )
    )

    assert result.safety_level == "CAUTION"
    assert result.data_as_of == NOW
    assert result.metadata["vehicleState"]["batteryTemperatureMaxC"] == 42.0
    assert result.metadata["vehicleState"]["cellVoltageMinV"] == 3.8


@pytest.mark.asyncio
async def test_rag_failure_logs_traceback_without_query_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_query = "private diagnostic question"

    with caplog.at_level(logging.ERROR, logger="ev-ai-chatbot"):
        result = await service(FakeRag(fail=True), FakeGenerator()).answer(
            ChatMessageRequest(message=private_query)
        )

    assert result.missing_fields == ["rag"]
    assert "RAG search failed for route=RAG" in caplog.text
    assert "RuntimeError: database down" in caplog.text
    assert private_query not in caplog.text
