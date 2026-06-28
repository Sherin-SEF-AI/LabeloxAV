"""LiDAR module Phase 2: 3D annotation. track_3d (a 3D track linked to the 2D track), object_3d (one cuboid,
linked to the 2D object by object_id, the unifying identity), and point_segmentation (per-point semantic and
instance labels). The same governed ontology and gate apply: class_id is an ontology class, conf is
calibrated, box_source records lifted vs native.

Revision ID: 0027_lidar_3d_annotation
Revises: 0026_lidar_pointcloud
Create Date: 2026-06-28
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision: str = "0027_lidar_3d_annotation"
down_revision: Union[str, None] = "0026_lidar_pointcloud"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "track_3d",
        sa.Column("track_3d_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("track_id", UUID(as_uuid=True), sa.ForeignKey("track.track_id", ondelete="SET NULL"), nullable=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("class_id", sa.Integer(), sa.ForeignKey("ontology_class.id"), nullable=False),
        sa.Column("first_ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("last_ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("trajectory", JSONB(), nullable=True),
        sa.Column("dynamic_state", sa.String(16), nullable=True),
    )
    op.create_index("ix_track_3d_session", "track_3d", ["session_id"])
    op.create_index("ix_track_3d_track", "track_3d", ["track_id"])

    op.create_table(
        "object_3d",
        sa.Column("object_3d_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("cloud_id", UUID(as_uuid=True), sa.ForeignKey("point_cloud.cloud_id", ondelete="CASCADE"), nullable=False),
        sa.Column("frame_id", UUID(as_uuid=True), sa.ForeignKey("frame.frame_id", ondelete="SET NULL"), nullable=True),
        sa.Column("object_id", UUID(as_uuid=True), sa.ForeignKey("object.object_id", ondelete="SET NULL"), nullable=True),
        sa.Column("track_3d_id", UUID(as_uuid=True), sa.ForeignKey("track_3d.track_3d_id", ondelete="SET NULL"), nullable=True),
        sa.Column("class_id", sa.Integer(), sa.ForeignKey("ontology_class.id"), nullable=False),
        sa.Column("center", ARRAY(sa.Float()), nullable=False),
        sa.Column("dims", ARRAY(sa.Float()), nullable=False),
        sa.Column("yaw", sa.Float(), nullable=False, server_default="0"),
        sa.Column("pitch", sa.Float(), nullable=False, server_default="0"),
        sa.Column("roll", sa.Float(), nullable=False, server_default="0"),
        sa.Column("conf", sa.Float(), nullable=False),
        sa.Column("box_source", sa.String(8), nullable=False),
        sa.Column("is_keyframe", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("interp_source", sa.String(16), nullable=True),
        sa.Column("source", sa.String(16), nullable=False, server_default="fused"),
        sa.Column("state", sa.String(16), nullable=False, server_default="review"),
        sa.Column("attrs", JSONB(), nullable=False, server_default="{}"),
        sa.Column("provenance", JSONB(), nullable=False, server_default="{}"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_object_3d_cloud", "object_3d", ["cloud_id"])
    op.create_index("ix_object_3d_frame", "object_3d", ["frame_id"])
    op.create_index("ix_object_3d_object", "object_3d", ["object_id"])
    op.create_index("ix_object_3d_track", "object_3d", ["track_3d_id"])

    op.create_table(
        "point_segmentation",
        sa.Column("seg_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("cloud_id", UUID(as_uuid=True), sa.ForeignKey("point_cloud.cloud_id", ondelete="CASCADE"), nullable=False),
        sa.Column("labels_uri", sa.Text(), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("method", sa.String(32), nullable=True),
        sa.Column("n_points", sa.Integer(), nullable=True),
        sa.Column("low_conf_frac", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_point_segmentation_cloud", "point_segmentation", ["cloud_id"])


def downgrade() -> None:
    op.drop_index("ix_point_segmentation_cloud", table_name="point_segmentation")
    op.drop_table("point_segmentation")
    for ix in ("ix_object_3d_track", "ix_object_3d_object", "ix_object_3d_frame", "ix_object_3d_cloud"):
        op.drop_index(ix, table_name="object_3d")
    op.drop_table("object_3d")
    op.drop_index("ix_track_3d_track", table_name="track_3d")
    op.drop_index("ix_track_3d_session", table_name="track_3d")
    op.drop_table("track_3d")
