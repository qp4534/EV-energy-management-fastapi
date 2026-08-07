from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )


class ReportType(StrEnum):
    ANOMALY = "ANOMALY"
    MONTHLY = "MONTHLY"

    @property
    def public_value(self) -> str:
        return "이상" if self is ReportType.ANOMALY else "월간보고서"


class ReportSource(ApiModel):
    chunk_id: str
    title: str
    source_type: str
    page: int | None = None
    clause: str | None = None
    url: str | None = None


class ReportSection(ApiModel):
    type: Literal["summary", "metricGrid", "lineChart", "numberedList", "bulletList"]
    title: str
    content: str | None = None
    items: list[Any] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    unit: str | None = None


class ReportPeriod(ApiModel):
    from_date: date = Field(alias="from")
    to_date: date = Field(alias="to")


class GeneratedReport(ApiModel):
    schema_version: str = "1.0"
    report_type: ReportType
    is_ai_generated: bool = True
    llm_enhanced: bool = False
    data_as_of: datetime
    risk_level: str
    period: ReportPeriod | None = None
    sections: list[ReportSection]
    sources: list[ReportSource] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)

    @field_serializer("report_type")
    def serialize_report_type(self, value: ReportType) -> str:
        return value.public_value


class NarrativeEnhancement(BaseModel):
    summary: str = Field(min_length=1, max_length=2_000)
    interpretation: str | None = Field(default=None, max_length=2_000)
    recommended_actions: list[str] = Field(
        default_factory=list, alias="recommendedActions", max_length=8
    )

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class MonthlyJobRequest(ApiModel):
    car_id: str = Field(min_length=1, max_length=128)
    target_month: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")


class JobResponse(ApiModel):
    job_id: str
    status: str
