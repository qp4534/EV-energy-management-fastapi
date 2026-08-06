from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        serialize_by_alias=True,
    )


class ChatMessageRequest(ApiModel):
    message: str = Field(min_length=1, max_length=4_000)
    user_id: str | None = Field(default=None, max_length=128)
    vehicle_id: str | None = Field(default=None, max_length=128)
    conversation_id: str | None = Field(default=None, max_length=128)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class SourceCitation(ApiModel):
    chunk_id: str
    title: str
    source_type: str
    page: int | None = None
    clause: str | None = None
    url: str | None = None
    score: float = Field(ge=0.0)


class ChatMessageResponse(ApiModel):
    answer: str
    route: str
    safety_level: str
    data_as_of: datetime | None = None
    sources: list[SourceCitation] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
