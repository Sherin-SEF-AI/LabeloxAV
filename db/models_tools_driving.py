"""Tables for the driving annotation tools (migration 0131_tools_driving).

Define classes on `db.models.Base` here and list them in `__all__`; db/models.py re-exports them.
"""

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
    String,
    Text,
    func,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.models import Base

__all__: list[str] = ["LaneRelation", "PathLabel", "CalibrationAlignment"]

LANE_RELATIONS = ("predecessor", "successor", "left_neighbour", "right_neighbour", "controlled_by", "stops_at")


class LaneRelation(Base):
    """One edge of lane topology a person drew: this lane line relates to exactly one target.

    Lane-to-lane: predecessor / successor (the line continues), left_neighbour / right_neighbour (the line on
    the other side of the same lane). Lane-to-regulation: controlled_by (a traffic signal, as an object box on
    the frame or a map element) and stops_at (a stop line, which is a lane row typed `stop_line`).
    """

    __tablename__ = "lane_relation"

    relation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    lane_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("lane.lane_id", ondelete="CASCADE"))
    relation: Mapped[str] = mapped_column(String(24), nullable=False)
    target_lane_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("lane.lane_id", ondelete="CASCADE"))
    target_object_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("object.object_id", ondelete="CASCADE"))
    target_element_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("map_element.element_id", ondelete="CASCADE"))
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("relation in ('predecessor','successor','left_neighbour','right_neighbour',"
                        "'controlled_by','stops_at')", name="ck_lane_relation_kind"),
        CheckConstraint("(target_lane_id is not null)::int + (target_object_id is not null)::int"
                        " + (target_element_id is not null)::int = 1", name="ck_lane_relation_one_target"),
        CheckConstraint("target_lane_id is null or target_lane_id <> lane_id", name="ck_lane_relation_not_self"),
        Index("ix_lane_relation_lane", "lane_id"),
        Index("ix_lane_relation_target_lane", "target_lane_id"),
        Index("ix_lane_relation_session", "session_id"),
    )


class PathLabel(Base):
    """A path a person labelled: the corrected ego path of a session, or a road user's future from one instant.

    `points` are [[t_s, x, y], ...]: seconds after `anchor_ts_ns` (negative allowed for the ego path, which
    spans the whole session), and metres in `coord_frame` - `world` is the session-local ENU frame of ego_pose,
    `ego` is the vehicle frame at the anchor instant (x forward, y left) when the session has no poses.
    """

    __tablename__ = "path_label"

    path_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    track_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("track.track_id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(16), nullable=False)            # ego_path | future
    anchor_frame_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("frame.frame_id", ondelete="SET NULL"))
    anchor_ts_ns: Mapped[int] = mapped_column(BigInteger, nullable=False)
    coord_frame: Mapped[str] = mapped_column(String(8), nullable=False)       # world | ego
    points: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    source: Mapped[str] = mapped_column(String(16), nullable=False)           # human | baseline_ctrv | smoothed
    seed: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)   # the baseline it started from
    notes: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 onupdate=func.now())

    __table_args__ = (
        CheckConstraint("kind in ('ego_path','future')", name="ck_path_label_kind"),
        CheckConstraint("coord_frame in ('world','ego')", name="ck_path_label_frame"),
        CheckConstraint("source in ('human','baseline_ctrv','smoothed')", name="ck_path_label_source"),
        CheckConstraint("(kind = 'ego_path') = (track_id is null)", name="ck_path_label_subject"),
        Index("ix_path_label_session", "session_id"),
        Index("ix_path_label_track", "track_id"),
        Index("ux_path_label_ego", "session_id", unique=True, postgresql_where=sql_text("kind = 'ego_path'")),
        Index("ux_path_label_future", "track_id", "anchor_ts_ns", unique=True,
              postgresql_where=sql_text("kind = 'future'")),
    )


class CalibrationAlignment(Base):
    """One camera-LiDAR alignment: where it started, where it ended, and the scores at both ends."""

    __tablename__ = "calibration_alignment"

    alignment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("session.session_id", ondelete="CASCADE"))
    cam_id: Mapped[str] = mapped_column(String(32), nullable=False)
    frame_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    method: Mapped[str] = mapped_column(String(16), nullable=False)           # manual | auto | pairs
    base_source: Mapped[str] = mapped_column(String(16), nullable=False)
    base_rpy_deg: Mapped[list] = mapped_column(JSONB, nullable=False)
    base_xyz_m: Mapped[list] = mapped_column(JSONB, nullable=False)
    rpy_deg: Mapped[list] = mapped_column(JSONB, nullable=False)
    xyz_m: Mapped[list] = mapped_column(JSONB, nullable=False)
    score_before: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    score_after: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    saved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=sql_text("false"))
    calibration_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("camera_calibration.id", ondelete="SET NULL"))
    tool_job_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    notes: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("app_user.user_id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("method in ('manual','auto','pairs')", name="ck_calibration_alignment_method"),
        Index("ix_calibration_alignment_session", "session_id", "cam_id"),
    )


# Indexes on the foreign keys (migration 0131), declared here so the metadata matches the schema.
Index("ix_calibration_alignment_calibration", CalibrationAlignment.calibration_id)
Index("ix_calibration_alignment_created_by", CalibrationAlignment.created_by)
Index("ix_lane_relation_created_by", LaneRelation.created_by)
Index("ix_lane_relation_target_element", LaneRelation.target_element_id)
Index("ix_lane_relation_target_object", LaneRelation.target_object_id)
Index("ix_path_label_anchor_frame", PathLabel.anchor_frame_id)
Index("ix_path_label_created_by", PathLabel.created_by)
