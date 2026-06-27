"""Phase 3 Multi-Sensor and Spatial schema: calibration_validation, camera_rig, map_element (geography),
map_commit, map_fusion_job; frame road context; object multi-camera columns

Revision ID: 0017_spatial
Revises: 0016_perception
Create Date: 2026-06-27
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geography
from sqlalchemy.dialects import postgresql

revision: str = "0017_spatial"
down_revision: Union[str, None] = "0016_perception"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # frame: map context (M3.2), all nullable
    op.add_column("frame", sa.Column("road_segment_id", sa.Text()))
    op.add_column("frame", sa.Column("road_class", sa.Text()))
    op.add_column("frame", sa.Column("lane_count", sa.Integer()))
    op.add_column("frame", sa.Column("speed_limit", sa.Integer()))

    # object: multi-camera (M3.1)
    op.add_column("object", sa.Column("rig_track_id", postgresql.UUID(as_uuid=True)))
    op.add_column("object", sa.Column("cross_cam_links", postgresql.JSONB()))

    op.create_table(
        "camera_rig",
        sa.Column("rig_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("vehicle_id", sa.String(64), nullable=False),
        sa.Column("cameras", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "calibration_validation",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("cam_id", sa.String(32), nullable=False),
        sa.Column("model", sa.String(16), nullable=False),
        sa.Column("reproj_error_px", sa.Float()),
        sa.Column("fov_check", postgresql.JSONB(), server_default="{}"),
        sa.Column("extrinsic_consistency", postgresql.JSONB()),
        sa.Column("time_offset_ns", sa.BigInteger()),
        sa.Column("status", sa.String(8), nullable=False, server_default="pass"),
        sa.Column("report_uri", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_calib_session", "calibration_validation", ["session_id"])

    op.create_table(
        "map_commit",
        sa.Column("commit_id", sa.String(128), primary_key=True),
        sa.Column("region", sa.String(128), nullable=False),
        sa.Column("session_ids", postgresql.ARRAY(sa.Text())),
        sa.Column("element_count", sa.Integer(), server_default="0"),
        sa.Column("formats", postgresql.JSONB(), server_default="{}"),
        sa.Column("calibration_version", sa.String(64)),
        sa.Column("fusion_job_id", postgresql.UUID(as_uuid=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "map_element",
        sa.Column("element_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("geometry", Geography(srid=4326)),
        sa.Column("attrs", postgresql.JSONB(), server_default="{}"),
        sa.Column("source_frames", postgresql.ARRAY(sa.Text())),
        sa.Column("source_sessions", postgresql.ARRAY(sa.Text())),
        sa.Column("calibration_version", sa.String(64)),
        sa.Column("confidence", sa.Float(), server_default="0"),
        sa.Column("fusion_job_id", postgresql.UUID(as_uuid=True)),
        sa.Column("commit_id", sa.String(128), sa.ForeignKey("map_commit.commit_id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_map_element_kind", "map_element", ["kind"])
    op.create_index("ix_map_element_commit", "map_element", ["commit_id"])

    op.create_table(
        "map_fusion_job",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("compute_target", sa.String(16), nullable=False, server_default="local"),
        sa.Column("region", sa.String(128), nullable=False),
        sa.Column("session_ids", postgresql.ARRAY(sa.Text())),
        sa.Column("stage", sa.String(24)),
        sa.Column("progress", sa.Float(), server_default="0"),
        sa.Column("counts", postgresql.JSONB(), server_default="{}"),
        sa.Column("result", postgresql.JSONB(), server_default="{}"),
        sa.Column("commit_id", sa.String(128)),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_map_fusion_status", "map_fusion_job", ["status"])


def downgrade() -> None:
    op.drop_table("map_fusion_job")
    op.drop_table("map_element")
    op.drop_table("map_commit")
    op.drop_table("calibration_validation")
    op.drop_table("camera_rig")
    op.drop_column("object", "cross_cam_links")
    op.drop_column("object", "rig_track_id")
    for c in ("speed_limit", "lane_count", "road_class", "road_segment_id"):
        op.drop_column("frame", c)
