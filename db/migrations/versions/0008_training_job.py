"""training_job queue + model_run registry generalization (in-app training platform, Phase 1)

Revision ID: 0008_training_job
Revises: 0007_import_job
Create Date: 2026-06-26
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_training_job"
down_revision: str | None = "0007_import_job"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "training_job",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("purpose", sa.String(64), nullable=False),
        sa.Column("task_type", sa.String(32), nullable=False, server_default="detection"),
        sa.Column("config", postgresql.JSONB(), server_default="{}"),
        sa.Column("stage", sa.String(24)),
        sa.Column("progress", sa.Float(), server_default="0"),
        sa.Column("counts", postgresql.JSONB(), server_default="{}"),
        sa.Column("metrics", postgresql.JSONB(), server_default="{}"),
        sa.Column("result", postgresql.JSONB(), server_default="{}"),
        sa.Column("error", sa.Text()),
        sa.Column("cancel_requested", sa.Boolean(), server_default=sa.false()),
        sa.Column("run_id", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_training_job_status", "training_job", ["status"])

    # Generalize model_run into a multi-line registry keyed by (purpose, task_type).
    op.add_column("model_run", sa.Column("purpose", sa.String(64), nullable=False, server_default="perception"))
    op.add_column("model_run", sa.Column("task_type", sa.String(32), nullable=False, server_default="detection"))
    op.add_column("model_run", sa.Column("job_id", postgresql.UUID(as_uuid=True)))


def downgrade() -> None:
    op.drop_column("model_run", "job_id")
    op.drop_column("model_run", "task_type")
    op.drop_column("model_run", "purpose")
    op.drop_index("ix_training_job_status", table_name="training_job")
    op.drop_table("training_job")
