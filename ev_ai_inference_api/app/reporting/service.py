from __future__ import annotations

import json
import re
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

KST = timezone(timedelta(hours=9))


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


def _metric(
    label: str,
    value: Any,
    unit: str | None = None,
    *,
    caption: str | None = None,
    emphasis: str | None = None,
) -> dict[str, Any]:
    result = {"label": label, "value": value}
    if unit:
        result["unit"] = unit
    if caption:
        result["caption"] = caption
    if emphasis:
        result["emphasis"] = emphasis
    return result


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _voltage_deviation_v(raw_metrics: dict[str, Any]) -> float | None:
    twin_frame = raw_metrics.get("twin_frame")
    if not isinstance(twin_frame, dict):
        return None
    values = twin_frame.get("voltage_mv")
    if not isinstance(values, list):
        return None
    numeric = [_number(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if len(numeric) < 2:
        return None
    return round((max(numeric) - min(numeric)) / 1_000.0, 3)


_VOLTAGE_DEVIATION_TRIGGER = re.compile(
    r"(?:전압\s*편차|voltage\s*(?:deviation|delta))\s*[:=]?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>m?v)\b",
    re.IGNORECASE,
)


def _trigger_voltage_deviation_v(facts: dict[str, Any]) -> float | None:
    abnormal_type = str(facts.get("abnormal_type") or "").strip().lower()
    is_voltage_anomaly = (
        "전압" in abnormal_type
        and ("편차" in abnormal_type or "불균형" in abnormal_type)
    ) or (
        "voltage" in abnormal_type
        and ("deviation" in abnormal_type or "imbalance" in abnormal_type)
    )
    if not is_voltage_anomaly:
        return None

    trigger_value = str(facts.get("trigger_value") or "").strip()
    match = _VOLTAGE_DEVIATION_TRIGGER.search(trigger_value)
    if match is None:
        return None

    value = float(match.group("value"))
    if match.group("unit").lower() == "mv":
        value /= 1_000.0
    if not 0.0 <= value <= 5.0:
        return None
    return round(value, 3)


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
        raw_metrics = facts.get("raw_metrics") or {}
        temperature = _number(model_input.get("temp_max_c"))
        temperature_caption = "이상 감지 시점 최고값"

        soh = _number(facts.get("soh_score"))
        previous_soh = _number(facts.get("previous_soh_score"))
        soh_delta = (
            round(soh - previous_soh, 1)
            if soh is not None and previous_soh is not None
            else None
        )
        cell_voltage_deviation = _voltage_deviation_v(raw_metrics)
        trigger_voltage_deviation = _trigger_voltage_deviation_v(facts)
        voltage_deviation = (
            cell_voltage_deviation
            if cell_voltage_deviation is not None
            else trigger_voltage_deviation
        )
        voltage_deviation_caption = (
            "96개 셀의 최대·최소 전압 차이"
            if cell_voltage_deviation is not None
            else (
                "이상 감지 로그 기준"
                if trigger_voltage_deviation is not None
                else "셀별 전압 데이터 없음"
            )
        )
        charge_cycles = facts.get("charge_cycles")
        detected_value = (
            detected_at.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S KST")
            if isinstance(detected_at, datetime)
            else str(detected_at)
        )
        risk_display = {
            "EMERGENCY": "긴급",
            "WARNING": "경고",
            "CAUTION": "주의",
            "NORMAL": "정상",
            "UNKNOWN": "미확인",
        }[risk]
        detection_metrics = [
            _metric("발생 시각", detected_value),
            _metric("위험등급", risk_display),
            _metric("이상 유형", abnormal_type),
        ]
        metrics = [
            _metric(
                "배터리 온도",
                round(temperature, 1) if temperature is not None else "확인 불가",
                "°C" if temperature is not None else None,
                caption=temperature_caption if temperature is not None else "측정 데이터 없음",
                emphasis=(
                    "danger"
                    if temperature is not None and temperature >= 45
                    else None
                ),
            ),
            _metric(
                "SOH 변동",
                f"{soh_delta:+.1f}" if soh_delta is not None else "확인 불가",
                "%p" if soh_delta is not None else None,
                caption=(
                    f"현재 SOH {soh:.1f}% · 이전 진단 대비"
                    if soh_delta is not None
                    else "비교 가능한 SOH 이력 없음"
                ),
            ),
            _metric(
                "전압 편차",
                voltage_deviation if voltage_deviation is not None else "확인 불가",
                "V" if voltage_deviation is not None else None,
                caption=voltage_deviation_caption,
            ),
            _metric(
                "누적 충전 사이클",
                charge_cycles if charge_cycles is not None else "확인 불가",
                "회" if charge_cycles is not None else None,
                caption="배터리 패스포트 기준",
            ),
        ]

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
        llm_facts["report_metrics"] = metrics
        enhancement = await self._enhance(llm_facts, chunks)
        sections = [
            ReportSection(type="summary", title="이상 상황 요약", content=summary),
            ReportSection(
                type="metricGrid",
                title="이상 감지 정보",
                items=detection_metrics,
            ),
            ReportSection(type="metricGrid", title="이상 지표", items=metrics),
        ]
        temperature_history = facts.get("temperature_history") or []
        chart_points = [
            item
            for item in temperature_history
            if isinstance(item, dict)
            and item.get("observed_at") is not None
            and _number(item.get("temperature_c")) is not None
        ]
        if chart_points:
            sections.append(
                ReportSection(
                    type="lineChart",
                    title="최근 24시간 온도 추이",
                    unit="°C",
                    labels=[
                        (
                            item["observed_at"].strftime("%m/%d %H시")
                            if isinstance(item["observed_at"], datetime)
                            else str(item["observed_at"])
                        )
                        for item in chart_points
                    ],
                    datasets=[
                        {
                            "label": "최고 배터리 온도",
                            "data": [
                                round(float(item["temperature_c"]), 1)
                                for item in chart_points
                            ],
                        }
                    ],
                )
            )
        if enhancement is not None:
            sections[0] = ReportSection(
                type="summary", title="이상 상황 요약", content=enhancement.summary
            )
            if enhancement.interpretation:
                sections.append(
                    ReportSection(
                        type="numberedList",
                        title="원인 분석",
                        items=[enhancement.interpretation],
                    )
                )
            if enhancement.recommended_actions:
                sections.append(
                    ReportSection(
                        type="bulletList",
                        title="권장 조치",
                        items=enhancement.recommended_actions,
                    )
                )
        missing: list[str] = []
        if facts.get("frame_observed_at") is None:
            missing.append("twinFrame")
        if not model_input:
            missing.append("modelInput")
        if soh_delta is None:
            missing.append("sohHistory")
        if cell_voltage_deviation is None:
            missing.append("cellVoltages")
        if charge_cycles is None:
            missing.append("chargeCycles")
        if not chart_points:
            missing.append("temperatureHistory")
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
        fleet = facts.get("fleet") or {}
        anomaly_count = sum(int(row.get("count") or 0) for row in anomaly_rows)
        risk = _highest_risk([row.get("risk_level") for row in anomaly_rows])
        total_duration_hours = round(
            float(sessions.get("total_duration_minutes") or 0) / 60.0,
            1,
        )
        report_sessions = dict(sessions)
        report_sessions.pop("total_duration_minutes", None)
        report_sessions["total_duration_hours"] = total_duration_hours

        metrics = [
            _metric("전체 차량", int(fleet.get("vehicle_count") or 0), "대"),
            _metric("충전 세션", int(sessions.get("session_count") or 0), "회"),
            _metric(
                "완료된 충전 세션",
                int(sessions.get("completed_session_count") or 0),
                "회",
            ),
            _metric(
                "총 충전 시간",
                total_duration_hours,
                "시간",
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
            "fleet": fleet,
            "periodStart": start,
            "periodEndExclusive": end_exclusive,
            "chargingSessions": report_sessions,
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
        return f"전체 차량 통합 월간 안전보고서 - {start:%Y-%m}", GeneratedReport(
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
