from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import text

from app.ai.config import AISettings
from app.db.session import create_database
from app.rag.embedding import SentenceTransformerEmbedder
from app.rag.repository import _vector_literal


_OFFICIAL_TYPES = {
    "official_form_summary",
    "official_law_summary",
    "official_law_full_text",
    "official_safety_guide",
}


@dataclass(frozen=True)
class NormalizedChunk:
    source_path: Path
    chunk_id: str
    document_id: str
    chunk_index: int | None
    collection: str
    source_title: str
    source_type: str
    authority: str | None
    jurisdiction: str | None
    effective_date: str | None
    current_as_of: str | None
    legal_status: str | None
    approved_for_deployment: bool
    content_hash: str
    content: str
    embedding_text: str
    metadata: dict[str, Any]


def _derived_document_id(record: dict[str, Any]) -> str:
    identity = "|".join(
        str(record.get(key) or "")
        for key in ("source_title", "source_type", "official_url", "authority")
    )
    return f"derived-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


def _approved(record: dict[str, Any]) -> bool:
    explicit = record.get("approved_for_deployment")
    if isinstance(explicit, bool):
        return explicit
    source_type = str(record.get("source_type") or "")
    if source_type in _OFFICIAL_TYPES:
        return True
    return bool(
        source_type == "technical_guide"
        and record.get("authority")
        and record.get("visibility") == "public"
    )


def normalize_record(
    record: dict[str, Any], source_path: Path, line_number: int
) -> NormalizedChunk:
    chunk_id = str(record.get("chunk_id") or "").strip()
    content = str(record.get("content") or "").strip()
    if not chunk_id:
        raise ValueError(f"{source_path}:{line_number}: chunk_id is required")
    if not content:
        raise ValueError(f"{source_path}:{line_number}: content is required")

    calculated_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    declared_hash = str(record.get("content_sha256") or "").strip().lower()
    if declared_hash and declared_hash != calculated_hash:
        raise ValueError(
            f"{source_path}:{line_number}: content_sha256 does not match content"
        )

    source_title = str(record.get("source_title") or source_path.stem).strip()
    source_type = str(record.get("source_type") or "unknown").strip()
    document_id = str(record.get("document_id") or "").strip()
    if not document_id:
        document_id = _derived_document_id(record)

    raw_index = record.get("chunk_index")
    chunk_index = None if raw_index is None else int(raw_index)
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"content", "embedding_text"}
    }
    embedding_text = str(record.get("embedding_text") or content).strip()

    return NormalizedChunk(
        source_path=source_path,
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        collection=str(record.get("collection") or "ev_safety_rag"),
        source_title=source_title,
        source_type=source_type,
        authority=(str(record["authority"]) if record.get("authority") else None),
        jurisdiction=(
            str(record["jurisdiction"]) if record.get("jurisdiction") else None
        ),
        effective_date=(
            str(record["effective_date"]) if record.get("effective_date") else None
        ),
        current_as_of=(
            str(record["current_as_of"]) if record.get("current_as_of") else None
        ),
        legal_status=(
            str(record["legal_status"]) if record.get("legal_status") else None
        ),
        approved_for_deployment=_approved(record),
        content_hash=calculated_hash,
        content=content,
        embedding_text=embedding_text,
        metadata=metadata,
    )


def iter_jsonl(source: Path) -> Iterable[NormalizedChunk]:
    paths = [source] if source.is_file() else sorted(source.rglob("*.jsonl"))
    if not paths:
        raise ValueError(f"no JSONL files found under {source}")
    seen_chunk_ids: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(raw, dict):
                    raise ValueError(
                        f"{path}:{line_number}: each line must be a JSON object"
                    )
                chunk = normalize_record(raw, path, line_number)
                if chunk.chunk_id in seen_chunk_ids:
                    raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
                seen_chunk_ids.add(chunk.chunk_id)
                yield chunk


def validation_summary(chunks: list[NormalizedChunk]) -> dict[str, Any]:
    return {
        "files": len({str(chunk.source_path) for chunk in chunks}),
        "documents": len({chunk.document_id for chunk in chunks}),
        "chunks": len(chunks),
        "approvedChunks": sum(chunk.approved_for_deployment for chunk in chunks),
        "sourceTypes": dict(sorted(Counter(chunk.source_type for chunk in chunks).items())),
    }


async def ingest(source: Path, settings: AISettings, *, validate_only: bool) -> dict[str, Any]:
    chunks = list(iter_jsonl(source))
    summary = validation_summary(chunks)
    if validate_only:
        return summary

    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
    )
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), settings.embedding_batch_size):
        batch = chunks[start : start + settings.embedding_batch_size]
        embeddings.extend(
            await embedder.embed_documents([chunk.embedding_text for chunk in batch])
        )

    engine, sessions = create_database(settings.database_url)
    try:
        document_groups: dict[str, list[NormalizedChunk]] = {}
        for chunk in chunks:
            document_groups.setdefault(chunk.document_id, []).append(chunk)

        async with sessions.begin() as session:
            for document_id, document_chunks in document_groups.items():
                first = document_chunks[0]
                manifest_hash = hashlib.sha256(
                    "".join(item.content_hash for item in document_chunks).encode("ascii")
                ).hexdigest()
                await session.execute(
                    text(
                        """
                        INSERT INTO rag_documents (
                            document_id, collection, source_title, source_type,
                            authority, jurisdiction, effective_date, current_as_of,
                            legal_status, approved_for_deployment, content_hash,
                            metadata, active, updated_at
                        ) VALUES (
                            :document_id, :collection, :source_title, :source_type,
                            :authority, :jurisdiction, CAST(:effective_date AS date),
                            CAST(:current_as_of AS date), :legal_status, :approved,
                            :content_hash, CAST(:metadata AS jsonb), TRUE, NOW()
                        )
                        ON CONFLICT (document_id) DO UPDATE SET
                            collection = EXCLUDED.collection,
                            source_title = EXCLUDED.source_title,
                            source_type = EXCLUDED.source_type,
                            authority = EXCLUDED.authority,
                            jurisdiction = EXCLUDED.jurisdiction,
                            effective_date = EXCLUDED.effective_date,
                            current_as_of = EXCLUDED.current_as_of,
                            legal_status = EXCLUDED.legal_status,
                            approved_for_deployment = EXCLUDED.approved_for_deployment,
                            content_hash = EXCLUDED.content_hash,
                            metadata = EXCLUDED.metadata,
                            active = TRUE,
                            updated_at = NOW()
                        """
                    ),
                    {
                        "document_id": document_id,
                        "collection": first.collection,
                        "source_title": first.source_title,
                        "source_type": first.source_type,
                        "authority": first.authority,
                        "jurisdiction": first.jurisdiction,
                        "effective_date": first.effective_date,
                        "current_as_of": first.current_as_of,
                        "legal_status": first.legal_status,
                        "approved": all(
                            item.approved_for_deployment for item in document_chunks
                        ),
                        "content_hash": manifest_hash,
                        "metadata": json.dumps(first.metadata, ensure_ascii=False),
                    },
                )
                await session.execute(
                    text(
                        "UPDATE rag_chunks SET active = FALSE WHERE document_id = :document_id"
                    ),
                    {"document_id": document_id},
                )

            chunk_statement = text(
                """
                INSERT INTO rag_chunks (
                    chunk_id, document_id, chunk_index, content, embedding,
                    metadata, content_hash, active, updated_at
                ) VALUES (
                    :chunk_id, :document_id, :chunk_index, :content,
                    CAST(:embedding AS vector), CAST(:metadata AS jsonb),
                    :content_hash, TRUE, NOW()
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    chunk_index = EXCLUDED.chunk_index,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    content_hash = EXCLUDED.content_hash,
                    active = TRUE,
                    updated_at = NOW()
                """
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                await session.execute(
                    chunk_statement,
                    {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "embedding": _vector_literal(
                            embedding, settings.embedding_dimension
                        ),
                        "metadata": json.dumps(chunk.metadata, ensure_ascii=False),
                        "content_hash": chunk.content_hash,
                    },
                )
        summary["status"] = "ingested"
        return summary
    finally:
        await engine.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or ingest RAG JSONL files into PostgreSQL/pgvector"
    )
    parser.add_argument("source", type=Path, help="JSONL file or directory")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate JSONL without loading embeddings or connecting to a database",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = asyncio.run(
        ingest(args.source.resolve(), AISettings.load(), validate_only=args.validate_only)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
