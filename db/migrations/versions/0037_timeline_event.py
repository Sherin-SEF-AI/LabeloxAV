"""timeline_event: human or auto events on the canonical session timeline (imu, audio, scene, geo,
crossmodal), with optimistic concurrency. source=auto events are unconfirmed candidates, never auto-accepted.

Revision ID: 0037_timeline_event
Revises: 0036_camera_calibration
Create Date: 2026-06-30
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0037_timeline_event"
down_revision: str | None = "0036_camera_calibration"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "timeline_event",
        sa.Column("event_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True),
                  sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(40), nullable=False),
        sa.Column("modality", sa.String(16), nullable=False),
        sa.Column("t_start_ns", sa.BigInteger(), nullable=False),
        sa.Column("t_end_ns", sa.BigInteger(), nullable=True),
        sa.Column("payload", JSONB(), server_default="{}"),
        sa.Column("source", sa.String(16), nullable=False, server_default="human"),
        sa.Column("state", sa.String(16), nullable=False, server_default="review"),
        sa.Column("provenance", JSONB(), server_default="{}"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_timeline_event_session_t", "timeline_event", ["session_id", "t_start_ns"])


def downgrade() -> None:
    op.drop_index("ix_timeline_event_session_t", table_name="timeline_event")
    op.drop_table("timeline_event")
