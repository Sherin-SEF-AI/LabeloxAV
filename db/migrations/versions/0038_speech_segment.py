"""speech_segment: detected human-speech regions, the third DPDPA modality alongside face and plate. A
personal un-redacted segment fail-closes the export gate.

Revision ID: 0038_speech_segment
Revises: 0037_timeline_event
Create Date: 2026-06-30
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0038_speech_segment"
down_revision: str | None = "0037_timeline_event"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "speech_segment",
        sa.Column("segment_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True),
                  sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("t_start_ns", sa.BigInteger(), nullable=False),
        sa.Column("t_end_ns", sa.BigInteger(), nullable=False),
        sa.Column("is_personal", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("redacted", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("method_version", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_speech_segment_session", "speech_segment", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_speech_segment_session", table_name="speech_segment")
    op.drop_table("speech_segment")
