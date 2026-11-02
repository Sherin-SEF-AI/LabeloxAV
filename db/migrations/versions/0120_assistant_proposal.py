"""What the assistant proposed, and what a person decided.

The assistant (services/assistant) reads freely but changes nothing on its own: every tool that would write
(reclassify, flag for review, auto-label a session, run error detection) records a proposal here with a
preview of what it would touch. A reviewer approves or rejects it; approval runs the change through the
same reversible, audited path a person would use, and the result is kept beside the proposal.

Revision ID: 0120_assistant_proposal
Revises: 0119_project_classes
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0120_assistant_proposal"
down_revision = "0119_project_classes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assistant_proposal",
        sa.Column("proposal_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tool", sa.String(64), nullable=False),
        sa.Column("args", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("preview", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("result", postgresql.JSONB(), nullable=True),
        sa.Column("request", sa.Text(), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("app_user.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("decided_by", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("app_user.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("status IN ('pending', 'approved', 'rejected', 'failed')",
                           name="ck_assistant_proposal_status"),
    )
    op.create_index("ix_assistant_proposal_status", "assistant_proposal", ["status", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_assistant_proposal_status", table_name="assistant_proposal")
    op.drop_table("assistant_proposal")
