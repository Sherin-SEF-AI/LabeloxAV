"""pii_audit: per-frame anonymization record (Gate A, DPDPA)

Revision ID: 0005_pii_audit
Revises: 0004_model_run
Create Date: 2026-06-26
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_pii_audit"
down_revision: str | None = "0004_model_run"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "pii_audit",
        sa.Column("frame_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE")),
        sa.Column("n_faces", sa.Integer(), server_default="0"),
        sa.Column("n_plates", sa.Integer(), server_default="0"),
        sa.Column("regions", postgresql.JSONB(), server_default="[]"),
        sa.Column("method_version", sa.String(64), nullable=False),
        sa.Column("ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_pii_audit_session", "pii_audit", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_pii_audit_session", table_name="pii_audit")
    op.drop_table("pii_audit")
