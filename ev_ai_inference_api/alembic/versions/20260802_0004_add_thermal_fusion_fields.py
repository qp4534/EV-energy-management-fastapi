"""Store optional synchronized thermal-model evidence on twin frames."""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260802_0004"
down_revision: str | None = "20260731_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("twin_frames", sa.Column("image_risk_level", sa.SmallInteger()))
    op.add_column("twin_frames", sa.Column("image_confidence", sa.Float()))
    op.add_column(
        "twin_frames",
        sa.Column("image_probabilities", sa.ARRAY(sa.Float())),
    )
    op.add_column(
        "twin_frames",
        sa.Column("image_model_status", sa.String(length=24)),
    )
    op.add_column(
        "twin_frames",
        sa.Column("module_heat_score", sa.ARRAY(sa.Float())),
    )
    op.add_column(
        "twin_frames",
        sa.Column("module_state_level", sa.ARRAY(sa.SmallInteger())),
    )
    op.add_column(
        "twin_frames",
        sa.Column("hotspot_module_index", sa.SmallInteger()),
    )
    op.add_column(
        "twin_frames",
        sa.Column("thermal_frame_ref", sa.String(length=512)),
    )
    op.add_column(
        "twin_frames",
        sa.Column("thermal_frame_sha256", sa.String(length=64)),
    )
    op.add_column("twin_frames", sa.Column("fusion_source", sa.String(length=24)))
    op.create_check_constraint(
        "ck_twin_frames_module_state_count",
        "twin_frames",
        "module_state_level IS NULL OR cardinality(module_state_level) = 12",
    )
    op.create_check_constraint(
        "ck_twin_frames_module_state_values",
        "twin_frames",
        "module_state_level IS NULL OR module_state_level <@ ARRAY[0,1,2,3]::smallint[]",
    )
    op.create_check_constraint(
        "ck_twin_frames_image_risk",
        "twin_frames",
        "image_risk_level IS NULL OR image_risk_level BETWEEN 0 AND 3",
    )
    op.create_check_constraint(
        "ck_twin_frames_hotspot_module",
        "twin_frames",
        "hotspot_module_index IS NULL OR hotspot_module_index BETWEEN 0 AND 11",
    )


def downgrade() -> None:
    for constraint in (
        "ck_twin_frames_hotspot_module",
        "ck_twin_frames_image_risk",
        "ck_twin_frames_module_state_values",
        "ck_twin_frames_module_state_count",
    ):
        op.drop_constraint(constraint, "twin_frames", type_="check")
    for column in (
        "fusion_source",
        "thermal_frame_sha256",
        "thermal_frame_ref",
        "hotspot_module_index",
        "module_state_level",
        "module_heat_score",
        "image_model_status",
        "image_probabilities",
        "image_confidence",
        "image_risk_level",
    ):
        op.drop_column("twin_frames", column)
