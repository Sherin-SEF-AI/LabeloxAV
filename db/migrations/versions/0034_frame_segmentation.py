"""frame_segmentation: full-frame dense segmentation (semantic class-id raster, optional panoptic
instance raster) with a colored display overlay, per-class coverage, panoptic segments, and lineage.

Revision ID: 0034_frame_segmentation
Revises: 0033_adverse_region
Create Date: 2026-06-29
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0034_frame_segmentation"
down_revision: str | None = "0033_adverse_region"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "frame_segmentation",
        sa.Column("seg_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("frame_id", UUID(as_uuid=True),
                  sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("labels_uri", sa.Text(), nullable=False),
        sa.Column("instance_uri", sa.Text(), nullable=True),
        sa.Column("overlay_uri", sa.Text(), nullable=True),
        sa.Column("coverage", JSONB(), nullable=False, server_default="{}"),
        sa.Column("segments", JSONB(), nullable=False, server_default="{}"),
        sa.Column("source", sa.String(16), nullable=False, server_default="proposed"),
        sa.Column("model_version", sa.String(64), nullable=True),
        sa.Column("ontology_version", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_frame_segmentation_frame_kind", "frame_segmentation", ["frame_id", "kind"])


def downgrade() -> None:
    op.drop_index("ix_frame_segmentation_frame_kind", table_name="frame_segmentation")
    op.drop_table("frame_segmentation")
