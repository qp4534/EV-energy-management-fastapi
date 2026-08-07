"""Store independent continuous heat scores for all 96 cells."""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260802_0005"
down_revision: str | None = "20260802_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "twin_frames",
        sa.Column("cell_heat_score", sa.ARRAY(sa.Float())),
    )
    op.create_check_constraint(
        "ck_twin_frames_cell_heat_count",
        "twin_frames",
        "cell_heat_score IS NULL OR cardinality(cell_heat_score) = 96",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_twin_frames_cell_heat_count",
        "twin_frames",
        type_="check",
    )
    op.drop_column("twin_frames", "cell_heat_score")
