from __future__ import annotations

import json
import math
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ai.config import AISettings
from app.ai.contracts import EmbeddingProvider, RetrievedChunk


_ROUTE_FILTERS = {
    "EMERGENCY": """
        AND d.source_type IN (
            'technical_guide', 'official_safety_guide',
            'emergency_safety_guide'
        )
    """,
    "LEGAL": """
        AND d.source_type IN (
            'official_law_summary', 'official_law_full_text',
            'official_form_summary'
        )
        AND (d.legal_status IS NULL OR d.legal_status = 'current')
    """,
    "VEHICLE_STATUS": """
        AND d.source_type IN ('glossary', 'safety_policy', 'user_faq')
    """,
    "REPORT": """
        AND d.source_type IN (
            'official_law_summary', 'official_law_full_text',
            'official_form_summary',
            'technical_guide', 'official_safety_guide',
            'emergency_safety_guide', 'safety_policy', 'glossary'
        )
    """,
    "RAG": "",
    "GENERAL": "",
}


def _vector_literal(vector: list[float], expected_dimension: int) -> str:
    if len(vector) != expected_dimension:
        raise ValueError(
            f"query embedding must contain {expected_dimension} dimensions"
        )
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("query embedding contains a non-finite value")
    return "[" + ",".join(format(float(value), ".9g") for value in vector) + "]"


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _optional_page(metadata: dict[str, Any]) -> int | None:
    value = metadata.get("source_page", metadata.get("page_number"))
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class PostgresRagRepository:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        embedder: EmbeddingProvider,
        settings: AISettings,
    ) -> None:
        self.sessions = sessions
        self.embedder = embedder
        self.settings = settings

    async def search(
        self,
        query: str,
        *,
        route: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        normalized_route = route.upper()
        if normalized_route not in _ROUTE_FILTERS:
            raise ValueError(f"unsupported RAG route: {route}")
        query = query.strip()
        if not query:
            return []

        result_limit = top_k or self.settings.rag_top_k
        if result_limit <= 0 or result_limit > 20:
            raise ValueError("top_k must be between 1 and 20")

        embedding = await self.embedder.embed_query(query)
        route_filter = _ROUTE_FILTERS[normalized_route]
        sql = text(
            f"""
            WITH vector_candidates AS (
                SELECT
                    c.chunk_id,
                    1 - (c.embedding <=> CAST(:embedding AS vector)) AS vector_score,
                    0.0::double precision AS keyword_score
                FROM rag_chunks c
                JOIN rag_documents d ON d.document_id = c.document_id
                WHERE c.active = TRUE
                  AND d.active = TRUE
                  AND c.embedding IS NOT NULL
                  AND (:allow_drafts OR d.approved_for_deployment = TRUE)
                  {route_filter}
                ORDER BY c.embedding <=> CAST(:embedding AS vector)
                LIMIT :candidate_k
            ),
            keyword_candidates AS (
                SELECT
                    c.chunk_id,
                    0.0::double precision AS vector_score,
                    ts_rank_cd(
                        c.search_vector,
                        plainto_tsquery('simple', :query)
                    )::double precision AS keyword_score
                FROM rag_chunks c
                JOIN rag_documents d ON d.document_id = c.document_id
                WHERE c.active = TRUE
                  AND d.active = TRUE
                  AND (:allow_drafts OR d.approved_for_deployment = TRUE)
                  AND c.search_vector @@ plainto_tsquery('simple', :query)
                  {route_filter}
                ORDER BY keyword_score DESC
                LIMIT :candidate_k
            ),
            combined AS (
                SELECT
                    chunk_id,
                    MAX(vector_score) AS vector_score,
                    MAX(keyword_score) AS keyword_score
                FROM (
                    SELECT * FROM vector_candidates
                    UNION ALL
                    SELECT * FROM keyword_candidates
                ) candidates
                GROUP BY chunk_id
            ),
            scored AS (
                SELECT
                    combined.chunk_id,
                    GREATEST(
                        combined.vector_score,
                        CASE
                            WHEN combined.keyword_score > 0
                            THEN 0.55 + LEAST(combined.keyword_score, 0.45)
                            ELSE 0
                        END
                    ) AS score
                FROM combined
            )
            SELECT
                c.chunk_id,
                c.document_id,
                d.source_title,
                d.source_type,
                c.content,
                c.metadata,
                scored.score
            FROM scored
            JOIN rag_chunks c ON c.chunk_id = scored.chunk_id
            JOIN rag_documents d ON d.document_id = c.document_id
            WHERE scored.score >= :min_score
            ORDER BY scored.score DESC, c.chunk_index NULLS LAST, c.chunk_id
            LIMIT :result_limit
            """
        )
        params = {
            "embedding": _vector_literal(
                embedding, self.settings.embedding_dimension
            ),
            "query": query,
            "allow_drafts": self.settings.rag_allow_drafts,
            "candidate_k": self.settings.rag_candidate_k,
            "min_score": self.settings.rag_min_score,
            "result_limit": result_limit,
        }
        async with self.sessions() as session:
            rows = (await session.execute(sql, params)).mappings().all()

        chunks: list[RetrievedChunk] = []
        for row in rows:
            metadata = _metadata(row["metadata"])
            chunks.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    source_title=row["source_title"],
                    source_type=row["source_type"],
                    content=row["content"],
                    score=float(row["score"]),
                    page=_optional_page(metadata),
                    clause=metadata.get("clause"),
                    official_url=metadata.get("official_url"),
                    metadata=metadata,
                )
            )
        return chunks
