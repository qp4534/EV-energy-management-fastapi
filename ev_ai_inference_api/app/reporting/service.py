from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol

from app.ai.contracts import RagRetriever, RetrievedChunk, TextGenerator

from .repository import ReportJob
from .schemas import (
    GeneratedReport,
    NarrativeEnhancement,
    ReportPeriod,
    ReportSection,
    ReportSource,
    ReportType,
)


class ReportDataRepository(Protocol):
    async def load_anomaly_facts(self, job: ReportJob) -> dict[str, Any]: ...

    async def load_monthly_facts(self, job: ReportJob) -> dict[str, Any]: ...


_REPORT_SYSTEM = """당신은 전기차 충전·배터리 안전보고서 문장 생성기다.
반드시 JSON 객체 하나만 반환한다. 키는 summary, interpretation,
recommendedActions 세 개만 사용한다.
- FACTS에 있는 사실과 EVIDENCE의 지침만 사용한다.
- 수치, 원인, 위험등급, 법령 조항을 만들지 않는다.
- 가능한 원인은 확정 원인이 아니라 가능성으로 표현한다.
- EVIDENCE가 뒷받침하지 않는 조치를 추가하지 않는다.
- 작성자, 검토자, 승인자, 서명, 고객센터 항목을 만들지 않는다.
"""


def _report_sources(chunks: list[RetrievedChunk]) -> list[ReportSource]:
    seen: set[tuple[str, int | None, str | None]] = set()
    result: list[ReportSource] = []
    for chunk in chunks:
        key = (chunk.document_id, chunk.page, chunk.clause)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            ReportSource(
                chunk_id=chunk.chunk_id,
                title=chunk.source_title,
                source_type=chunk.source_type,
                page=chunk.page,
                clause=chunk.clause,
                url=chunk.official_url,
            )
        )
    return result


def _evidence(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[SOURCE_{index}] {chunk.source_title}\n{chunk.content[:1800]}"
        for index, chunk in enumerate(chunks, start=1)
    )


def _risk_label(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"3", "emergency", "긴급", "위험", "심각"}:
        return "EMERGENCY"
    if normalized in {"2", "warning", "경고"}:
        return "WARNING"
    if normalized in {"1", "caution", "주의"}:
        return "CAUTION"
    if normalized in {"0", "normal", "정상"}:
        return "NORMAL"
    return "UNKNOWN"


def _highest_risk(values: list[Any]) -> str:
    order = {"UNKNOWN": -1, "NORMAL": 0, "CAUTION": 1, "WARNING": 2, "EMERGENCY": 3}
    labels = [_risk_label(value) for value in values]
    return max(labels, key=order.__getitem__, default="NORMAL")


def _metric(label: str, value: Any, unit: str | None = None) -> dict[str, Any]:
    result = {"label": label, "value": value}
    if unit:
        result["unit"] = unit
    return result


class ReportGenerationService:
    def __init__(
        self,
        repository: ReportDataRepository,
        rag: RagRetriever,
        generator: TextGenerator,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.rag = rag
        self.generator = generator
        self.now = now or (lambda: datetime.now(timezone.utc))

    async def generate(self, job: ReportJob) -> tuple[str, GeneratedReport]:
        if job.job_type == ReportType.ANOMALY:
            facts = await self.repository.load_anomaly_facts(job)
            return await self._anomaly(facts)
        facts = await self.repository.load_monthly_facts(job)
        return await self._monthly(facts)

    async def _search(self, query: str) -> list[RetrievedChunk]:
        try:
            return await self.rag.search(query, route="REPORT")
        except Exception:
            return []

    async def _enhance(
        self, facts: dict[str, Any], chunks: list[RetrievedChunk]
    ) -> NarrativeEnhancement | None:
        if not chunks:
            return None
        prompt = (
            "아래 FACTS와 EVIDENCE로 보고서 설명을 작성하고 JSON으로 반환하라.\n\n"
            "FACTS:\n"
            + json.dumps(facts, ensure_ascii=False, default=str)
            + "\n\nEVIDENCE:\n"
            + _evidence(chunks)
            + '\n\nJSON 예시: {"summary":"...","interpretation":"...",'
            '"recommendedActions":["..."]}'
        )
        try:
            raw = await self.generator.generate(
                _REPORT_SYSTEM,
                prompt,
                purpose="report",
                json_mode=True,
            )
            return NarrativeEnhancement.model_validate_json(raw)
        except Exception:
            return None

    async def _anomaly(self, facts: dict[str, Any]) -> tuple[str, GeneratedReport]:
        detected_at = facts.get("detected_at") or self.now()
        if isinstance(detected_at, str):
            detected_at = datetime.fromisoformat(detected_at.replace("Z", "+00:00"))
        risk = _risk_label(facts.get("risk_level") or facts.get("final_risk_level"))
        abnormal_type = facts.get("abnormal_type") or "분류되지 않은 이상"
        model_input = facts.get("model_input") or {}
        metrics = [
            _metric("발생 시각", detected_at.isoformat() if isinstance(detected_at, datetime) else str(detected_at)),
            _metric("위험등급", risk),
            _metric("이상 유형", abnormal_type),
        ]
        known_fields = (
            ("temp_max_c", "최고 배터리 온도", "°C"),
            ("temp_mean_c", "평균 배터리 온도", "°C"),
            ("temp_delta_c", "배터리 온도 편차", "°C"),
            ("voltage_v", "평균 셀 전압", "V"),
        )
        for key, label, unit in known_fields:
            if model_input.get(key) is not None:
                metrics.append(_metric(label, model_input[key], unit))

        summary = f"{detected_at:%Y-%m-%d %H:%M}에 {abnormal_type} 이상이 감지되었습니다."
        query = f"{abnormal_type} 위험등급 {risk} 안전 조치 전기차 배터리 충전"
        chunks = await self._search(query)
        llm_facts = {
            key: facts.get(key)
            for key in (
                "abnormal_type",
                "source_type",
                "trigger_value",
                "detected_at",
                "risk_level",
                "frame_observed_at",
                "hotspot_cell_index",
                "hotspot_connector_index",
                "ml_risk_level",
                "physics_risk_level",
                "final_risk_level",
                "image_risk_level",
                "image_confidence",
                "model_input",
            )
        }
        enhancement = await self._enhance(llm_facts, chunks)
        sections = [
            ReportSection(type="summary", title="이상 상황 요약", content=summary),
            ReportSection(type="metricGrid", title="감지 데이터", items=metrics),
        ]
        if enhancement is not None:
            sections[0] = ReportSection(
                type="summary", title="이상 상황 요약", content=enhancement.summary
            )
            if enhancement.interpretation:
                sections.append(
                    ReportSection(
                        type="summary",
                        title="데이터 해석",
                        content=enhancement.interpretation,
                    )
                )
            if enhancement.recommended_actions:
                sections.append(
                    ReportSection(
                        type="bulletList",
                        title="권장 안전 조치",
                        items=enhancement.recommended_actions,
                    )
                )
        missing: list[str] = []
        if facts.get("frame_observed_at") is None:
            missing.append("twinFrame")
        if not model_input:
            missing.append("modelInput")
        if not chunks:
            missing.append("ragEvidence")
        title = f"이상 보고서 - {detected_at:%Y-%m-%d %H:%M}"
        return title, GeneratedReport(
            report_type=ReportType.ANOMALY,
            llm_enhanced=enhancement is not None,
            data_as_of=self.now(),
            risk_level=risk,
            sections=sections,
            sources=_report_sources(chunks),
            missing_fields=missing,
            actions=[],
        )

    async def _monthly(self, facts: dict[str, Any]) -> tuple[str, GeneratedReport]:
        start: date = facts["periodStart"]
        end_exclusive: date = facts["periodEndExclusive"]
        sessions = facts.get("chargingSessions") or {}
        anomaly_rows = facts.get("anomalies") or []
        sensor = facts.get("sensorSummary") or {}
        anomaly_count = sum(int(row.get("count") or 0) for row in anomaly_rows)
        risk = _highest_risk([row.get("risk_level") for row in anomaly_rows])

        metrics = [
            _metric("충전 세션", int(sessions.get("session_count") or 0), "회"),
            _metric(
                "완료된 충전 세션",
                int(sessions.get("completed_session_count") or 0),
                "회",
            ),
            _metric(
                "총 충전 시간",
                round(float(sessions.get("total_duration_minutes") or 0), 1),
                "분",
            ),
            _metric("이상 발생", anomaly_count, "건"),
        ]
        if sessions.get("average_soc_change") is not None:
            metrics.append(
                _metric(
                    "평균 SOC 변화",
                    round(float(sessions["average_soc_change"]), 1),
                    "%p",
                )
            )
        if sensor.get("highest_temperature_c") is not None:
            metrics.append(
                _metric(
                    "월간 최고 배터리 온도",
                    round(float(sensor["highest_temperature_c"]), 1),
                    "°C",
                )
            )
        if sensor.get("average_max_temperature_c") is not None:
            metrics.append(
                _metric(
                    "평균 최고 배터리 온도",
                    round(float(sensor["average_max_temperature_c"]), 1),
                    "°C",
                )
            )

        summary = (
            f"{start:%Y년 %m월}에는 충전 세션 "
            f"{int(sessions.get('session_count') or 0)}회와 이상 {anomaly_count}건이 기록되었습니다."
        )
        chunks = await self._search(
            f"전기차 월간 안전 점검 위험등급 {risk} 충전 배터리 권장 조치"
        )
        llm_facts = {
            "periodStart": start,
            "periodEndExclusive": end_exclusive,
            "chargingSessions": sessions,
            "anomalies": anomaly_rows,
            "sensorSummary": sensor,
        }
        enhancement = await self._enhance(llm_facts, chunks)
        sections = [
            ReportSection(type="summary", title="월간 요약", content=summary),
            ReportSection(type="metricGrid", title="주요 지표", items=metrics),
        ]
        if anomaly_rows:
            sections.append(
                ReportSection(
                    type="bulletList",
                    title="위험등급별 이상 현황",
                    items=[
                        f"{_risk_label(row.get('risk_level'))}: {int(row.get('count') or 0)}건"
                        for row in anomaly_rows
                    ],
                )
            )
        if enhancement is not None:
            sections[0] = ReportSection(
                type="summary", title="월간 요약", content=enhancement.summary
            )
            if enhancement.interpretation:
                sections.append(
                    ReportSection(
                        type="summary",
                        title="월간 데이터 해석",
                        content=enhancement.interpretation,
                    )
                )
            if enhancement.recommended_actions:
                sections.append(
                    ReportSection(
                        type="bulletList",
                        title="다음 달 안전 권장사항",
                        items=enhancement.recommended_actions,
                    )
                )

        missing: list[str] = []
        if int(sensor.get("sample_count") or 0) == 0:
            missing.append("sensorSamples")
        if sessions.get("average_soc_change") is None:
            missing.append("averageSocChange")
        if not chunks:
            missing.append("ragEvidence")
        return f"월간 안전 보고서 - {start:%Y-%m}", GeneratedReport(
            report_type=ReportType.MONTHLY,
            llm_enhanced=enhancement is not None,
            data_as_of=self.now(),
            risk_level=risk,
            period=ReportPeriod(from_date=start, to_date=end_exclusive - timedelta(days=1)),
            sections=sections,
            sources=_report_sources(chunks),
            missing_fields=missing,
            actions=[],
        )
