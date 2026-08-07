"""Allow incidents with a short or gapped three-hour window to close incomplete."""

from typing import Sequence

from alembic import op


revision: str = "20260731_0002"
down_revision: str | None = "20260731_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_twin_incidents_status",
        "twin_incidents",
        type_="check",
    )
    op.create_check_constraint(
        "ck_twin_incidents_status",
        "twin_incidents",
        "status IN (0, 1, 2)",
    )


def downgrade() -> None:
    op.execute("UPDATE twin_incidents SET status = 0 WHERE status = 2")
    op.drop_constraint(
        "ck_twin_incidents_status",
        "twin_incidents",
        type_="check",
    )
    op.create_check_constraint(
        "ck_twin_incidents_status",
        "twin_incidents",
        "status IN (0, 1)",
    )
