"""autolabel_job: UI-triggered autolabel sweeps (operational layer)

Revision ID: 0010_autolabel_job
Revises: 0009_compute_target
Create Date: 2026-06-27
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010_autolabel_job"
down_revision: str | None = "0009_compute_target"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "autolabel_job",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True)),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("progress", sa.Float(), server_default="0"),
        sa.Column("counts", postgresql.JSONB(), server_default="{}"),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_autolabel_job_status", "autolabel_job", ["status"])


def downgrade() -> None:
    op.drop_index("ix_autolabel_job_status", table_name="autolabel_job")
    op.drop_table("autolabel_job")
