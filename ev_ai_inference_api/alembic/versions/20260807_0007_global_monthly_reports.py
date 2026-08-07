"""Allow fleet-wide monthly reports without a vehicle id."""

from typing import Sequence

from alembic import op


revision: str = "20260807_0007"
down_revision: str | None = "20260806_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute('ALTER TABLE public."AI_REPORTS" ALTER COLUMN car_id DROP NOT NULL')
    op.execute("ALTER TABLE ai_report_jobs ALTER COLUMN car_id DROP NOT NULL")


def downgrade() -> None:
    op.execute("DELETE FROM ai_report_jobs WHERE car_id IS NULL")
    op.execute('DELETE FROM public."AI_REPORTS" WHERE car_id IS NULL')
    op.execute("ALTER TABLE ai_report_jobs ALTER COLUMN car_id SET NOT NULL")
    op.execute('ALTER TABLE public."AI_REPORTS" ALTER COLUMN car_id SET NOT NULL')
