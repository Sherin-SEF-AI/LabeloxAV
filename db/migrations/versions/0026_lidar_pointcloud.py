"""LiDAR module Phase 1: point_cloud (one row per scan or synthesized cloud), point_cloud_derived (cleaned
and ground-removed variants, raw never overwritten), and lidar_calibration_validation (the LiDAR triple).
Clouds link to camera frames by session_id and the PPS ts_ns.

Revision ID: 0026_lidar_pointcloud
Revises: 0025_keypoints
Create Date: 2026-06-28
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0026_lidar_pointcloud"
down_revision: str | None = "0025_keypoints"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "point_cloud",
        sa.Column("cloud_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("cloud_uri", sa.Text(), nullable=False),
        sa.Column("point_count", sa.Integer(), nullable=False),
        sa.Column("depth_model", sa.String(96), nullable=True),
        sa.Column("calibration_version", sa.String(64), nullable=True),
        sa.Column("bounds", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_point_cloud_session_ts", "point_cloud", ["session_id", "ts_ns"])

    op.create_table(
        "point_cloud_derived",
        sa.Column("derived_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("cloud_id", UUID(as_uuid=True), sa.ForeignKey("point_cloud.cloud_id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(24), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("method", sa.String(48), nullable=False),
        sa.Column("params", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_point_cloud_derived_cloud", "point_cloud_derived", ["cloud_id"])

    op.create_table(
        "lidar_calibration_validation",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("pair", sa.String(24), nullable=False),
        sa.Column("reproj_error", sa.Float(), nullable=True),
        sa.Column("consistency", JSONB(), nullable=False, server_default="{}"),
        sa.Column("drift_flag", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(8), nullable=False, server_default="pass"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_lidar_calib_session", "lidar_calibration_validation", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_lidar_calib_session", table_name="lidar_calibration_validation")
    op.drop_table("lidar_calibration_validation")
    op.drop_index("ix_point_cloud_derived_cloud", table_name="point_cloud_derived")
    op.drop_table("point_cloud_derived")
    op.drop_index("ix_point_cloud_session_ts", table_name="point_cloud")
    op.drop_table("point_cloud")
