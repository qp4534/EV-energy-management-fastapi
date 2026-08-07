from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    SmallInteger,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


INCIDENT_OPEN = 0
INCIDENT_COMPLETE = 1
INCIDENT_INCOMPLETE = 2
INCIDENT_TYPE_GENERAL = "general"
INCIDENT_TYPE_CONNECTOR = "connector"
INCIDENT_TYPE_BATTERY = "battery"
INCIDENT_TYPES = frozenset(
    {INCIDENT_TYPE_GENERAL, INCIDENT_TYPE_CONNECTOR, INCIDENT_TYPE_BATTERY}
)


class Base(DeclarativeBase):
    pass


class TwinIncident(Base):
    __tablename__ = "twin_incidents"
    __table_args__ = (
        CheckConstraint("status IN (0, 1, 2)", name="ck_twin_incidents_status"),
        CheckConstraint(
            "incident_type IN ('general', 'connector', 'battery')",
            name="ck_twin_incidents_type",
        ),
        CheckConstraint(
            "window_start < triggered_at AND triggered_at < window_end",
            name="ck_twin_incidents_window",
        ),
        UniqueConstraint(
            "vehicle_id",
            "triggered_at",
            name="uq_twin_incidents_vehicle_triggered_at",
        ),
        Index(
            "ix_twin_incidents_vehicle_triggered_desc",
            "vehicle_id",
            "triggered_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    vehicle_id: Mapped[str] = mapped_column(String(128), nullable=False)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    incident_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default=INCIDENT_TYPE_GENERAL
    )
    status: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=INCIDENT_OPEN
    )
    rearmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    frames: Mapped[list["TwinFrameRecord"]] = relationship(
        back_populates="incident",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class TwinFrameRecord(Base):
    __tablename__ = "twin_frames"
    __table_args__ = (
        CheckConstraint(
            "layout_id = 'generic_ev_concept_96_v1'",
            name="ck_twin_frames_layout",
        ),
        CheckConstraint(
            "cardinality(temperature_decic) = 96",
            name="ck_twin_frames_temperature_count",
        ),
        CheckConstraint(
            "cardinality(voltage_mv) = 96",
            name="ck_twin_frames_voltage_count",
        ),
        CheckConstraint(
            "cardinality(state_level) = 96",
            name="ck_twin_frames_state_count",
        ),
        CheckConstraint(
            "cardinality(connector_temperature_decic) = 3",
            name="ck_twin_frames_connector_temperature_count",
        ),
        CheckConstraint(
            "cardinality(connector_state_level) = 3",
            name="ck_twin_frames_connector_state_count",
        ),
        CheckConstraint(
            "state_level <@ ARRAY[0,1,2,3]::smallint[]",
            name="ck_twin_frames_state_values",
        ),
        CheckConstraint(
            "connector_state_level <@ ARRAY[0,1,2,3]::smallint[]",
            name="ck_twin_frames_connector_state_values",
        ),
        CheckConstraint(
            "hotspot_cell_index BETWEEN 0 AND 95",
            name="ck_twin_frames_hotspot_cell",
        ),
        CheckConstraint(
            "hotspot_connector_index BETWEEN 0 AND 2",
            name="ck_twin_frames_hotspot_connector",
        ),
        CheckConstraint(
            "ml_risk_level IS NULL OR ml_risk_level BETWEEN 0 AND 3",
            name="ck_twin_frames_ml_risk",
        ),
        CheckConstraint(
            "physics_risk_level BETWEEN 0 AND 3",
            name="ck_twin_frames_physics_risk",
        ),
        CheckConstraint(
            "final_risk_level BETWEEN 0 AND 3",
            name="ck_twin_frames_final_risk",
        ),
        CheckConstraint(
            "module_state_level IS NULL OR cardinality(module_state_level) = 12",
            name="ck_twin_frames_module_state_count",
        ),
        CheckConstraint(
            "module_state_level IS NULL OR module_state_level <@ ARRAY[0,1,2,3]::smallint[]",
            name="ck_twin_frames_module_state_values",
        ),
        CheckConstraint(
            "image_risk_level IS NULL OR image_risk_level BETWEEN 0 AND 3",
            name="ck_twin_frames_image_risk",
        ),
        CheckConstraint(
            "hotspot_module_index IS NULL OR hotspot_module_index BETWEEN 0 AND 11",
            name="ck_twin_frames_hotspot_module",
        ),
        CheckConstraint(
            "cell_heat_score IS NULL OR cardinality(cell_heat_score) = 96",
            name="ck_twin_frames_cell_heat_count",
        ),
        UniqueConstraint(
            "incident_id",
            "observed_at",
            name="uq_twin_frames_incident_observed_at",
        ),
        Index("ix_twin_frames_incident_observed", "incident_id", "observed_at"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(), primary_key=True, autoincrement=True
    )
    incident_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("twin_incidents.id", ondelete="CASCADE"),
        nullable=False,
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    schema_version: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    layout_id: Mapped[str] = mapped_column(String(64), nullable=False)
    temperature_decic: Mapped[list[int]] = mapped_column(
        ARRAY(SmallInteger), nullable=False
    )
    voltage_mv: Mapped[list[int]] = mapped_column(ARRAY(SmallInteger), nullable=False)
    state_level: Mapped[list[int]] = mapped_column(ARRAY(SmallInteger), nullable=False)
    connector_temperature_decic: Mapped[list[int]] = mapped_column(
        ARRAY(SmallInteger), nullable=False
    )
    connector_state_level: Mapped[list[int]] = mapped_column(
        ARRAY(SmallInteger), nullable=False
    )
    hotspot_cell_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    hotspot_connector_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    ml_risk_level: Mapped[int | None] = mapped_column(SmallInteger)
    physics_risk_level: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    final_risk_level: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    cell_heat_score: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
    image_risk_level: Mapped[int | None] = mapped_column(SmallInteger)
    image_confidence: Mapped[float | None] = mapped_column()
    image_probabilities: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
    image_model_status: Mapped[str | None] = mapped_column(String(24))
    module_heat_score: Mapped[list[float] | None] = mapped_column(ARRAY(Float))
    module_state_level: Mapped[list[int] | None] = mapped_column(ARRAY(SmallInteger))
    hotspot_module_index: Mapped[int | None] = mapped_column(SmallInteger)
    thermal_frame_ref: Mapped[str | None] = mapped_column(String(512))
    thermal_frame_sha256: Mapped[str | None] = mapped_column(String(64))
    fusion_source: Mapped[str | None] = mapped_column(String(24))

    incident: Mapped[TwinIncident] = relationship(back_populates="frames")
