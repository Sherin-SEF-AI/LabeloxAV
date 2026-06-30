"""curation_slice: a named, persisted dataset cohort (predicate over scene axes + class/state/geo/conf) that
feeds export, training, and review.

Revision ID: 0039_curation_slice
Revises: 0038_speech_segment
Create Date: 2026-06-30
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0039_curation_slice"
down_revision: str | None = "0038_speech_segment"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "curation_slice",
        sa.Column("slice_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(80), nullable=False, unique=True),
        sa.Column("description", sa.String(240), nullable=True),
        sa.Column("predicate", JSONB(), nullable=False, server_default="{}"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("curation_slice")
