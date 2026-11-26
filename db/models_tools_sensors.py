"""Tables for the sensors tools (migration 0132_tools_sensors): radar sweeps, event streams, reconstructions."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.models import Base

__all__: list[str] = ["EventStream", "RadarSweep", "Reconstruction"]


class RadarSweep(Base):
    """One radar scan. The arrays (ego-frame positions, compensated velocity, radial velocity, RCS) are an npz in
    the object store at `uri`; `sensor_to_ego` is the 4x4 it was placed with, kept so a re-import can be checked."""

    __tablename__ = "radar_sweep"

    sweep_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sensor: Mapped[str] = mapped_column(String(48), nullable=False)        # RADAR_FRONT, /radar/front...
    source: Mapped[str] = mapped_column(String(24), nullable=False)        # nuscenes | mcap_pc2 | mcap_radarscan | pcd
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    n_points: Mapped[int] = mapped_column(Integer, nullable=False)
    has_velocity: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=sql_text("false"))
    sensor_to_ego: Mapped[list | None] = mapped_column(JSONB)
    stats: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=sql_text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_radar_sweep_session_ts", "session_id", "ts_ns"),
        Index("ux_radar_sweep_sensor_ts", "session_id", "sensor", "ts_ns", unique=True),
        CheckConstraint("n_points >= 0", name="ck_radar_sweep_points"),
    )


class EventStream(Base):
    """One event camera recording: raw events as an npz at `uri`, accumulated on demand into ordinary frames."""

    __tablename__ = "event_stream"

    stream_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    sensor: Mapped[str] = mapped_column(String(32), nullable=False)        # the cam_id its frames get
    source: Mapped[str] = mapped_column(String(24), nullable=False)        # npy | h5 | mcap
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    n_events: Mapped[int] = mapped_column(BigInteger, nullable=False)
    t0_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    t1_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    stats: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=sql_text("'{}'::jsonb"))
    accumulation: Mapped[dict | None] = mapped_column(JSONB)               # the last accumulation's params + counts
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_event_stream_session", "session_id"),
        CheckConstraint("t1_ns >= t0_ns", name="ck_event_stream_span"),
    )


class Reconstruction(Base):
    """A session window rebuilt in 3D. `artifacts` holds object-store uris (splats, .ply, preview cloud),
    `views` the rendered views with their cameras, `metrics` the fit (PSNR on held-out frames)."""

    __tablename__ = "reconstruction"

    recon_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    method: Mapped[str] = mapped_column(String(16), nullable=False)        # gsplat | pointmap
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", server_default="pending")
    cam_id: Mapped[str | None] = mapped_column(String(32))
    first_ts_ns: Mapped[int | None] = mapped_column(BigInteger)
    last_ts_ns: Mapped[int | None] = mapped_column(BigInteger)
    params: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=sql_text("'{}'::jsonb"))
    artifacts: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=sql_text("'{}'::jsonb"))
    views: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default=sql_text("'[]'::jsonb"))
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default=sql_text("'{}'::jsonb"))
    error: Mapped[str | None] = mapped_column(Text)
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tool_job.job_id", ondelete="SET NULL"))
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_reconstruction_session", "session_id", "created_at"),
        Index("ix_reconstruction_job", "job_id"),
        CheckConstraint("method in ('gsplat','pointmap')", name="ck_reconstruction_method"),
        CheckConstraint("status in ('pending','running','done','error','canceled')", name="ck_reconstruction_status"),
    )
