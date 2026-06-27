"""frame_embedding: whole-frame DINOv2 features for active-learning curation

Revision ID: 0013_frame_embedding
Revises: 0012_export_job
Create Date: 2026-06-27
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0013_frame_embedding"
down_revision: Union[str, None] = "0012_export_job"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "frame_embedding",
        sa.Column("frame_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("model", sa.String(48), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("vec", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("frame_embedding")
