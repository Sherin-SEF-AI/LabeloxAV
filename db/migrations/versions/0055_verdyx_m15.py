"""VERDYX M15: safety + statistical eval. A safety metrics blob (track/scenario metrics, bootstrap CIs,
significance) on each evaluation. Additive only.

Revision ID: 0055_verdyx_m15
Revises: 0054_oraclyx_m14
Create Date: 2026-07-06
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0055_verdyx_m15"
down_revision: str | None = "0054_oraclyx_m14"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("evaluation", sa.Column("safety", postgresql.JSONB(astext_type=sa.Text()),
                                          server_default=sa.text("'{}'::jsonb"), nullable=False))


def downgrade() -> None:
    op.drop_column("evaluation", "safety")
