"""Classify persisted incidents for connector and battery history demos."""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260731_0003"
down_revision: str | None = "20260731_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "twin_incidents",
        sa.Column(
            "incident_type",
            sa.String(length=16),
            nullable=False,
            server_default=sa.text("'general'"),
        ),
    )
    op.create_check_constraint(
        "ck_twin_incidents_type",
        "twin_incidents",
        "incident_type IN ('general', 'connector', 'battery')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_twin_incidents_type",
        "twin_incidents",
        type_="check",
    )
    op.drop_column("twin_incidents", "incident_type")
