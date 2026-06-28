"""import_job: multi-format dataset import jobs (Deliverable 3)

Revision ID: 0007_import_job
Revises: 0006_gold_set
Create Date: 2026-06-26
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_import_job"
down_revision: str | None = "0006_gold_set"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "import_job",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("format", sa.String(32), nullable=False),
        sa.Column("source_uri", sa.Text()),
        sa.Column("target_vehicle", sa.String(64), nullable=False),
        sa.Column("city", sa.String(64)),
        sa.Column("progress", sa.Float(), server_default="0"),
        sa.Column("counts", postgresql.JSONB(), server_default="{}"),
        sa.Column("error", sa.Text()),
        sa.Column("session_id", postgresql.UUID(as_uuid=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_import_job_status", "import_job", ["status"])


def downgrade() -> None:
    op.drop_index("ix_import_job_status", table_name="import_job")
    op.drop_table("import_job")
