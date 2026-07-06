"""LabeloxAV M13: label quality layer. A per-annotation quality/agreement/audit side table. Additive only.

Revision ID: 0053_labelox_m13
Revises: 0052_sievyx_m12
Create Date: 2026-07-06
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0053_labelox_m13"
down_revision: str | None = "0052_sievyx_m12"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "annotation_quality",
        sa.Column("object_id", UUID(as_uuid=True), sa.ForeignKey("object.object_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("quality", sa.Float(), nullable=False, server_default="0"),
        sa.Column("agreement", sa.Float(), nullable=True),
        sa.Column("flags", JSONB(), nullable=False, server_default="[]"),
        sa.Column("audit_verdict", sa.String(12), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("annotation_quality")
