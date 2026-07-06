"""Flywheel M17: adaptive controller. A ledger of adaptive cycles (label-budget allocation + collection tasks
driven by VERDYX failures and SIEVYX ODD gaps). Additive only.

Revision ID: 0057_flywheel_m17
Revises: 0056_forgyx_m16
Create Date: 2026-07-06
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0057_flywheel_m17"
down_revision: str | None = "0056_forgyx_m16"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "flywheel_cycle",
        sa.Column("cycle_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("label_budget", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("signals", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("allocation", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("collection_tasks", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("flywheel_cycle")
