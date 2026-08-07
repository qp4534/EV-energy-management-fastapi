"""Add pgvector RAG storage and durable AI report jobs."""

from typing import Sequence

from alembic import op


revision: str = "20260806_0006"
down_revision: str | None = "20260802_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE rag_documents (
            document_id text PRIMARY KEY,
            collection text NOT NULL,
            source_title text NOT NULL,
            source_type text NOT NULL,
            authority text,
            jurisdiction text,
            effective_date date,
            current_as_of date,
            legal_status text,
            approved_for_deployment boolean NOT NULL DEFAULT false,
            content_hash character(64) NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
            active boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE rag_chunks (
            chunk_id text PRIMARY KEY,
            document_id text NOT NULL
                REFERENCES rag_documents(document_id) ON DELETE CASCADE,
            chunk_index integer,
            content text NOT NULL,
            embedding vector(768),
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
            content_hash character(64) NOT NULL,
            search_vector tsvector GENERATED ALWAYS AS (
                to_tsvector('simple', coalesce(content, ''))
            ) STORED,
            active boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX ix_rag_chunks_document ON rag_chunks(document_id, chunk_index)"
    )
    op.execute(
        "CREATE INDEX ix_rag_chunks_search ON rag_chunks USING gin(search_vector)"
    )
    op.execute(
        """
        CREATE INDEX ix_rag_chunks_embedding_cosine
        ON rag_chunks USING hnsw (embedding vector_cosine_ops)
        WHERE active = true AND embedding IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE TABLE ai_report_jobs (
            job_id uuid PRIMARY KEY,
            job_key text NOT NULL UNIQUE,
            job_type text NOT NULL
                CHECK (job_type IN ('ANOMALY', 'MONTHLY')),
            car_id uuid NOT NULL,
            anomaly_id uuid,
            target_month date,
            status text NOT NULL DEFAULT 'PENDING'
                CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
            retry_count integer NOT NULL DEFAULT 0
                CHECK (retry_count >= 0),
            available_at timestamptz NOT NULL DEFAULT now(),
            started_at timestamptz,
            completed_at timestamptz,
            report_id uuid,
            error_message text,
            payload jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT ck_ai_report_jobs_target CHECK (
                (job_type = 'ANOMALY' AND anomaly_id IS NOT NULL AND target_month IS NULL)
                OR
                (job_type = 'MONTHLY' AND anomaly_id IS NULL AND target_month IS NOT NULL)
            )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX ix_ai_report_jobs_claim
        ON ai_report_jobs(status, available_at, created_at)
        WHERE status = 'PENDING'
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ai_report_jobs")
    op.execute("DROP TABLE IF EXISTS rag_chunks")
    op.execute("DROP TABLE IF EXISTS rag_documents")
    # The vector extension may be shared by other applications; keep it installed.
