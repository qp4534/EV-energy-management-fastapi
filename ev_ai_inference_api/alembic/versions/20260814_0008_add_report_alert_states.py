"""Track per-vehicle alert incidents for idempotent AI report jobs."""

from typing import Sequence

from alembic import op


revision: str = "20260814_0008"
down_revision: str | None = "20260807_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE ai_report_alert_states (
            car_id uuid PRIMARY KEY,
            incident_id uuid,
            current_risk_level smallint NOT NULL DEFAULT 0
                CHECK (current_risk_level BETWEEN 0 AND 3),
            last_reported_risk_level smallint NOT NULL DEFAULT 0
                CHECK (last_reported_risk_level BETWEEN 0 AND 3),
            candidate_risk_level smallint
                CHECK (candidate_risk_level BETWEEN 1 AND 2),
            candidate_count integer NOT NULL DEFAULT 0
                CHECK (candidate_count >= 0),
            normal_since timestamptz,
            last_observed_at timestamptz NOT NULL,
            created_at timestamptz NOT NULL DEFAULT NOW(),
            updated_at timestamptz NOT NULL DEFAULT NOW(),
            CONSTRAINT ck_ai_report_alert_states_incident CHECK (
                incident_id IS NOT NULL
                OR (current_risk_level = 0 AND last_reported_risk_level = 0)
            )
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ai_report_alert_states")
