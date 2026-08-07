import logging
from datetime import datetime, timedelta, timezone

import pytest

from app.ai.config import AISettings
from app.ai.contracts import RetrievedChunk
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


def service(rag, generator, vehicle=None, **setting_updates) -> ChatbotService:
    return ChatbotService(
        ChatSupervisor(),
        rag,
        generator,
        vehicle or FakeVehicleState(),
        settings(**setting_updates),
        now=lambda: NOW,
    )


def test_supervisor_uses_safety_first_deterministic_routes() -> None:
    supervisor = ChatSupervisor()
    assert supervisor.classify("충전기에서 연기가 나요") == ChatRoute.EMERGENCY
    assert supervisor.classify("한국전기설비규정 조항 알려줘") == ChatRoute.LEGAL
    assert supervisor.classify("지금 내 차 배터리 상태는?") == ChatRoute.VEHICLE_STATUS
    assert supervisor.classify("안녕하세요") == ChatRoute.GENERAL
    assert supervisor.classify("충전기는 어떻게 사용해?") == ChatRoute.RAG


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
