"""M-F.1 label quality score: a per-object composite QA signal, distinct from confidence, stored so the
review queue can rank by it and exports can carry it.

Revision ID: 0046_quality_score
Revises: 0045_multicam
Create Date: 2026-07-03
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0046_quality_score"
down_revision: str | None = "0045_multicam"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("object", sa.Column("quality_score", sa.Float(), nullable=True))
    op.create_index("ix_object_quality_score", "object", ["quality_score"])


def downgrade() -> None:
    op.drop_index("ix_object_quality_score", table_name="object")
    op.drop_column("object", "quality_score")
