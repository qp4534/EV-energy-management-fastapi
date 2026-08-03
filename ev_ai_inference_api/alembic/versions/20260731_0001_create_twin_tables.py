"""Create digital-twin incident and frame tables."""

from typing import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260731_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "twin_incidents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("vehicle_id", sa.String(length=128), nullable=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.SmallInteger(), nullable=False),
        sa.Column("rearmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("status IN (0, 1)", name="ck_twin_incidents_status"),
        sa.CheckConstraint(
            "window_start < triggered_at AND triggered_at < window_end",
            name="ck_twin_incidents_window",
        ),
        sa.UniqueConstraint(
            "vehicle_id",
            "triggered_at",
            name="uq_twin_incidents_vehicle_triggered_at",
        ),
    )
    op.create_index(
        "ix_twin_incidents_vehicle_triggered_desc",
        "twin_incidents",
        ["vehicle_id", "triggered_at"],
        unique=False,
    )
    op.create_table(
        "twin_frames",
        sa.Column(
            "id",
            sa.BigInteger(),
            sa.Identity(always=False),
            nullable=False,
        ),
        sa.Column("incident_id", sa.String(length=36), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("schema_version", sa.SmallInteger(), nullable=False),
        sa.Column("layout_id", sa.String(length=64), nullable=False),
        sa.Column(
            "temperature_decic",
            postgresql.ARRAY(sa.SmallInteger()),
            nullable=False,
        ),
        sa.Column(
            "voltage_mv", postgresql.ARRAY(sa.SmallInteger()), nullable=False
        ),
        sa.Column(
            "state_level", postgresql.ARRAY(sa.SmallInteger()), nullable=False
        ),
        sa.Column(
            "connector_temperature_decic",
            postgresql.ARRAY(sa.SmallInteger()),
            nullable=False,
        ),
        sa.Column(
            "connector_state_level",
            postgresql.ARRAY(sa.SmallInteger()),
            nullable=False,
        ),
        sa.Column("hotspot_cell_index", sa.SmallInteger(), nullable=False),
        sa.Column("hotspot_connector_index", sa.SmallInteger(), nullable=False),
        sa.Column("ml_risk_level", sa.SmallInteger(), nullable=True),
        sa.Column("physics_risk_level", sa.SmallInteger(), nullable=False),
        sa.Column("final_risk_level", sa.SmallInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["incident_id"], ["twin_incidents.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "layout_id = 'generic_ev_concept_96_v1'",
            name="ck_twin_frames_layout",
        ),
        sa.CheckConstraint(
            "cardinality(temperature_decic) = 96",
            name="ck_twin_frames_temperature_count",
        ),
        sa.CheckConstraint(
            "cardinality(voltage_mv) = 96",
            name="ck_twin_frames_voltage_count",
        ),
        sa.CheckConstraint(
            "cardinality(state_level) = 96",
            name="ck_twin_frames_state_count",
        ),
        sa.CheckConstraint(
            "cardinality(connector_temperature_decic) = 3",
            name="ck_twin_frames_connector_temperature_count",
        ),
        sa.CheckConstraint(
            "cardinality(connector_state_level) = 3",
            name="ck_twin_frames_connector_state_count",
        ),
        sa.CheckConstraint(
            "state_level <@ ARRAY[0,1,2,3]::smallint[]",
            name="ck_twin_frames_state_values",
        ),
        sa.CheckConstraint(
            "connector_state_level <@ ARRAY[0,1,2,3]::smallint[]",
            name="ck_twin_frames_connector_state_values",
        ),
        sa.CheckConstraint(
            "hotspot_cell_index BETWEEN 0 AND 95",
            name="ck_twin_frames_hotspot_cell",
        ),
        sa.CheckConstraint(
            "hotspot_connector_index BETWEEN 0 AND 2",
            name="ck_twin_frames_hotspot_connector",
        ),
        sa.CheckConstraint(
            "ml_risk_level IS NULL OR ml_risk_level BETWEEN 0 AND 3",
            name="ck_twin_frames_ml_risk",
        ),
        sa.CheckConstraint(
            "physics_risk_level BETWEEN 0 AND 3",
            name="ck_twin_frames_physics_risk",
        ),
        sa.CheckConstraint(
            "final_risk_level BETWEEN 0 AND 3",
            name="ck_twin_frames_final_risk",
        ),
        sa.UniqueConstraint(
            "incident_id",
            "observed_at",
            name="uq_twin_frames_incident_observed_at",
        ),
    )
    op.create_index(
        "ix_twin_frames_incident_observed",
        "twin_frames",
        ["incident_id", "observed_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_twin_frames_incident_observed", table_name="twin_frames")
    op.drop_table("twin_frames")
    op.drop_index(
        "ix_twin_incidents_vehicle_triggered_desc", table_name="twin_incidents"
    )
    op.drop_table("twin_incidents")
