"""Widen agent_run.kind from 16 to 32 chars. The agent fleet introduces longer run kinds
(overnight_auditor, drift_investigator, annotation_copilot, ...) that overflow the original 16-char column.

Revision ID: 0041_agent_run_kind_widen
Revises: 0040_agent_run
Create Date: 2026-07-02
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0041_agent_run_kind_widen"
down_revision: str | None = "0040_agent_run"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("agent_run", "kind", type_=sa.String(32), existing_nullable=False)


def downgrade() -> None:
    op.alter_column("agent_run", "kind", type_=sa.String(16), existing_nullable=False)
