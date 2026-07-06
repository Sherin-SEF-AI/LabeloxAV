"""Hardening M19: per-plane SLO observability ledger. Additive only.

Revision ID: 0059_hardening_m19
Revises: 0058_govern_m18
Create Date: 2026-07-06
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0059_hardening_m19"
down_revision: str | None = "0058_govern_m18"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "plane_slo",
        sa.Column("slo_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("plane", sa.String(length=16), nullable=False),
        sa.Column("window_s", sa.Float(), nullable=False, server_default="0"),
        sa.Column("met", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("breaches", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_plane_slo_plane", "plane_slo", ["plane", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_plane_slo_plane", table_name="plane_slo")
    op.drop_table("plane_slo")
