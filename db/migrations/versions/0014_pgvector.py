"""pgvector extension for the Data Intelligence Layer embedding HNSW indexes

Revision ID: 0014_pgvector
Revises: 0013_frame_embedding
Create Date: 2026-06-27
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0014_pgvector"
down_revision: Union[str, None] = "0013_frame_embedding"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    # Leave the extension installed; other objects may depend on it.
    pass
