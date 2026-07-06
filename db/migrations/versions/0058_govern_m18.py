"""Governance M18: PII redaction proof, consent + retention records. Additive only.

Revision ID: 0058_govern_m18
Revises: 0057_flywheel_m17
Create Date: 2026-07-06
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0058_govern_m18"
down_revision: str | None = "0057_flywheel_m17"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "redaction_proof",
        sa.Column("proof_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("release_commit", sa.String(length=128), nullable=False),
        sa.Column("n_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("n_covered", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("coverage", sa.Float(), nullable=False, server_default="0"),
        sa.Column("verdict", sa.String(length=12), nullable=False, server_default="fail"),
        sa.Column("signature", sa.String(length=128), nullable=True),
        sa.Column("uncovered", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_redaction_proof_release", "redaction_proof", ["release_commit"])
    op.create_table(
        "consent_record",
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("session.session_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("consent_status", sa.String(length=12), nullable=False, server_default="unknown"),
        sa.Column("legal_basis", sa.String(length=64), nullable=True),
        sa.Column("retention_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("consent_record")
    op.drop_index("ix_redaction_proof_release", table_name="redaction_proof")
    op.drop_table("redaction_proof")
