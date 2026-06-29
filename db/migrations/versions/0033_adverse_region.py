"""adverse_region: frame-level polygon regions tagged with an adverse condition (glare, reflection,
shadow, rain, fog, lowlight), so downstream models know which pixels to distrust.

Revision ID: 0033_adverse_region
Revises: 0032_object_relationship
Create Date: 2026-06-29
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0033_adverse_region"
down_revision: str | None = "0032_object_relationship"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "adverse_region",
        sa.Column("region_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("frame_id", UUID(as_uuid=True),
                  sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False),
        sa.Column("geometry", JSONB(), nullable=False),
        sa.Column("condition", sa.String(16), nullable=False),
        sa.Column("source", sa.String(16), nullable=False, server_default="human"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_adverse_region_frame", "adverse_region", ["frame_id"])


def downgrade() -> None:
    op.drop_index("ix_adverse_region_frame", table_name="adverse_region")
    op.drop_table("adverse_region")
