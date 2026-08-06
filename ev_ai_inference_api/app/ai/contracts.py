from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source_title: str
    source_type: str
    content: str
    score: float
    page: int | None = None
    clause: str | None = None
    official_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EmbeddingProvider(Protocol):
    dimension: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    async def embed_query(self, text: str) -> list[float]:
        ...


class RagRetriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        route: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        ...


class TextGenerator(Protocol):
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        purpose: Literal["chat", "report"],
        json_mode: bool = False,
    ) -> str:
        ...


class VehicleStateProvider(Protocol):
    async def latest(self, vehicle_id: str) -> dict[str, Any] | None:
        ...
