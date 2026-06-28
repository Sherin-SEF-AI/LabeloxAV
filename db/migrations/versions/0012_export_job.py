"""export_job: UI-driven async dataset exports (delivery surface)

Revision ID: 0012_export_job
Revises: 0011_users
Create Date: 2026-06-27
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012_export_job"
down_revision: str | None = "0011_users"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "export_job",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("progress", sa.Float(), server_default="0"),
        sa.Column("spec", postgresql.JSONB(), server_default="{}"),
        sa.Column("commit_id", sa.String(128)),
        sa.Column("object_count", sa.Integer(), server_default="0"),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_export_job_status", "export_job", ["status"])


def downgrade() -> None:
    op.drop_index("ix_export_job_status", table_name="export_job")
    op.drop_table("export_job")
