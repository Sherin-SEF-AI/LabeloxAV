"""agent_run: an auditable, reversible unit of autonomous annotation-agent work. Records the policy a run
applied, its per-object state transitions (so a run reverts exactly), the critic's findings, and roll-up
counts. This is the guardrail that makes agent auto-accept safe at scale.

Revision ID: 0040_agent_run
Revises: 0039_curation_slice
Create Date: 2026-07-01
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0040_agent_run"
down_revision: str | None = "0039_curation_slice"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_run",
        sa.Column("run_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False),  # frame|session|flywheel
        sa.Column("scope", JSONB(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(16), nullable=False, server_default="planned"),  # planned|committed|reverted|error
        sa.Column("policy", JSONB(), nullable=False, server_default="{}"),
        sa.Column("counts", JSONB(), nullable=False, server_default="{}"),
        sa.Column("changes", JSONB(), nullable=False, server_default="{}"),
        sa.Column("critic", JSONB(), nullable=False, server_default="{}"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("reverted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_agent_run_status", "agent_run", ["status"])
    op.create_index("ix_agent_run_kind", "agent_run", ["kind"])


def downgrade() -> None:
    op.drop_index("ix_agent_run_kind", table_name="agent_run")
    op.drop_index("ix_agent_run_status", table_name="agent_run")
    op.drop_table("agent_run")
