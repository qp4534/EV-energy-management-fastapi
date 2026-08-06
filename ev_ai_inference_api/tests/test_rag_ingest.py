import hashlib
import json
from datetime import date

import pytest

from app.rag.ingest import iter_jsonl, normalize_record, validation_summary


def test_normalizer_accepts_official_law_schema_without_document_id(tmp_path) -> None:
    path = tmp_path / "official.jsonl"
    content = "공식 발췌 내용"
    record = {
        "chunk_id": "law-1",
        "collection": "legal_kr",
        "source_type": "official_law_summary",
        "source_title": "전기안전 관련 고시",
        "authority": "관계부처",
        "official_url": "https://example.invalid/law",
        "content": content,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    first = list(iter_jsonl(path))[0]

    assert first.document_id.startswith("derived-")
    assert first.embedding_text == content
    assert first.approved_for_deployment is True
    assert validation_summary([first])["chunks"] == 1


def test_normalizer_keeps_internal_draft_unapproved(tmp_path) -> None:
    record = {
        "chunk_id": "draft-1",
        "document_id": "draft-doc",
        "source_type": "user_faq",
        "source_title": "FAQ",
        "approved_for_deployment": False,
        "content": "초안",
    }

    chunk = normalize_record(record, tmp_path / "draft.jsonl", 1)

    assert chunk.approved_for_deployment is False


def test_normalizer_parses_postgres_date_fields(tmp_path) -> None:
    record = {
        "chunk_id": "dated-1",
        "document_id": "dated-doc",
        "source_type": "glossary",
        "source_title": "용어사전",
        "effective_date": "2026-08-04",
        "current_as_of": "2026-08-06",
        "content": "날짜가 있는 문서",
    }

    chunk = normalize_record(record, tmp_path / "dated.jsonl", 1)

    assert chunk.effective_date == date(2026, 8, 4)
    assert chunk.current_as_of == date(2026, 8, 6)


def test_normalizer_rejects_invalid_date(tmp_path) -> None:
    record = {
        "chunk_id": "bad-date-1",
        "source_type": "glossary",
        "source_title": "용어사전",
        "effective_date": "2026-02-30",
        "content": "잘못된 날짜",
    }

    with pytest.raises(ValueError, match="effective_date must be YYYY-MM-DD"):
        normalize_record(record, tmp_path / "bad-date.jsonl", 1)


def test_normalizer_rejects_hash_mismatch(tmp_path) -> None:
    record = {
        "chunk_id": "bad-1",
        "source_type": "user_faq",
        "source_title": "FAQ",
        "content": "실제 내용",
        "content_sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="does not match"):
        normalize_record(record, tmp_path / "bad.jsonl", 1)


def test_iter_jsonl_rejects_duplicate_chunk_ids(tmp_path) -> None:
    path = tmp_path / "duplicate.jsonl"
    row = {"chunk_id": "same", "source_title": "FAQ", "content": "내용"}
    path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n" + json.dumps(row, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate chunk_id"):
        list(iter_jsonl(path))
