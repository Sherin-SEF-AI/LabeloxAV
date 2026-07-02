"""promotion_proposal: the Ontology Steward's evidence packet for a fallback cluster proposed as a new class,
awaiting one-click approve/reject.

Revision ID: 0042_promotion_proposal
Revises: 0041_agent_run_kind_widen
Create Date: 2026-07-02
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0042_promotion_proposal"
down_revision: str | None = "0041_agent_run_kind_widen"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "promotion_proposal",
        sa.Column("proposal_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("from_class", sa.Integer(), nullable=False),
        sa.Column("member_count", sa.Integer(), nullable=False),
        sa.Column("rep_object_ids", JSONB(), nullable=False, server_default="[]"),
        sa.Column("suggested_name", sa.String(64), nullable=True),
        sa.Column("confusion_classes", JSONB(), nullable=False, server_default="[]"),
        sa.Column("evidence_uri", sa.Text(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="proposed"),
        sa.Column("approved_class", sa.Integer(), nullable=True),
        sa.Column("run_id", UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_promotion_proposal_status", "promotion_proposal", ["status"])


def downgrade() -> None:
    op.drop_index("ix_promotion_proposal_status", table_name="promotion_proposal")
    op.drop_table("promotion_proposal")
