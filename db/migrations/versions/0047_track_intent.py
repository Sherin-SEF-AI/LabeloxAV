"""M-F.2 behavior and intent annotation: track-level typed intent from a closed vocabulary, proposed by
trajectory or VLM and confirmed by a human. Stored as a list on the track.

Revision ID: 0047_track_intent
Revises: 0046_quality_score
Create Date: 2026-07-03
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0047_track_intent"
down_revision: str | None = "0046_quality_score"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("track", sa.Column("intents", JSONB(), nullable=False, server_default="[]"))


def downgrade() -> None:
    op.drop_column("track", "intents")
