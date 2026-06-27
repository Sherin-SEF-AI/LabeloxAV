"""object embeddings (CLIP semantic search)

Revision ID: 0003_embedding
Revises: 0002_scenario
Create Date: 2026-06-25
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_embedding"
down_revision: Union[str, None] = "0002_scenario"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "embedding",
        sa.Column("object_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("object.object_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("model", sa.String(48), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("vec", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("embedding")
