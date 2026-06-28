"""LiDAR module Phase 3: 3D scene intelligence and export. static_element (extracted HD map candidates),
traversability (free space, drivable, surface, elevation), aggregated_map (registered multi-scan map),
quality_flag_3d (detected 3D label problems), and additive 3D counts on dataset_commit.

Revision ID: 0028_lidar_scene_intelligence
Revises: 0027_lidar_3d_annotation
Create Date: 2026-06-29
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geography
from sqlalchemy.dialects.postgresql import ARRAY as PGARRAY
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0028_lidar_scene_intelligence"
down_revision: Union[str, None] = "0027_lidar_3d_annotation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "static_element",
        sa.Column("element_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("geometry", Geography(srid=4326), nullable=True),
        sa.Column("attrs", JSONB(), nullable=False, server_default="{}"),
        sa.Column("source_clouds", PGARRAY(UUID(as_uuid=True)), nullable=True),
        sa.Column("method", sa.String(40), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("calibration_version", sa.String(64), nullable=True),
        sa.Column("map_element_id", UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_static_element_session", "static_element", ["session_id"])
    op.create_index("ix_static_element_kind", "static_element", ["kind"])

    op.create_table(
        "traversability",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("cloud_id", UUID(as_uuid=True), sa.ForeignKey("point_cloud.cloud_id", ondelete="CASCADE"), nullable=True),
        sa.Column("tile_id", sa.String(64), nullable=True),
        sa.Column("freespace_uri", sa.Text(), nullable=True),
        sa.Column("drivable_uri", sa.Text(), nullable=True),
        sa.Column("surface_class", JSONB(), nullable=False, server_default="{}"),
        sa.Column("elevation_profile", JSONB(), nullable=False, server_default="{}"),
        sa.Column("method", sa.String(40), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_traversability_cloud", "traversability", ["cloud_id"])

    op.create_table(
        "aggregated_map",
        sa.Column("agg_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("region", sa.String(64), nullable=True),
        sa.Column("session_ids", PGARRAY(UUID(as_uuid=True)), nullable=True),
        sa.Column("cloud_uri", sa.Text(), nullable=True),
        sa.Column("pose_graph", JSONB(), nullable=False, server_default="{}"),
        sa.Column("loop_closures", JSONB(), nullable=False, server_default="{}"),
        sa.Column("method", sa.String(40), nullable=True),
        sa.Column("n_scans", sa.Integer(), nullable=True),
        sa.Column("mean_reg_fitness", sa.Float(), nullable=True),
        sa.Column("job_id", UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_aggregated_map_region", "aggregated_map", ["region"])

    op.create_table(
        "quality_flag_3d",
        sa.Column("flag_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("object_3d_id", UUID(as_uuid=True), sa.ForeignKey("object_3d.object_3d_id", ondelete="CASCADE"), nullable=True),
        sa.Column("cloud_id", UUID(as_uuid=True), sa.ForeignKey("point_cloud.cloud_id", ondelete="CASCADE"), nullable=True),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("detail", JSONB(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_quality_flag_3d_object", "quality_flag_3d", ["object_3d_id"])
    op.create_index("ix_quality_flag_3d_status", "quality_flag_3d", ["status"])

    op.add_column("dataset_commit", sa.Column("object_3d_count", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("dataset_commit", sa.Column("cloud_count", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("dataset_commit", "cloud_count")
    op.drop_column("dataset_commit", "object_3d_count")
    for ix in ("ix_quality_flag_3d_status", "ix_quality_flag_3d_object"):
        op.drop_index(ix, table_name="quality_flag_3d")
    op.drop_table("quality_flag_3d")
    op.drop_index("ix_aggregated_map_region", table_name="aggregated_map")
    op.drop_table("aggregated_map")
    op.drop_index("ix_traversability_cloud", table_name="traversability")
    op.drop_table("traversability")
    for ix in ("ix_static_element_kind", "ix_static_element_session"):
        op.drop_index(ix, table_name="static_element")
    op.drop_table("static_element")
