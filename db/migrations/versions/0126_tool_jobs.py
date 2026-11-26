"""One queue for the annotation tools' background work.

Every job kind so far got its own table (import, export, autolabel, training, map fusion, relabel), each with
its own claim, cancel and listing code. The annotation tools add a dozen more kinds (SAM 2 video tracking,
session-wide text prompts, 3D propagation, occupancy, reconstruction...), and a table each would be a dozen
copies of the same columns. This is the one table they share: `kind` says which function runs it, `params`
what it was asked, `result` what it produced.

Revision ID: 0126_tool_jobs
Revises: 0125_llm_usage
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0126_tool_jobs"
down_revision = "0125_llm_usage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tool_job",
        sa.Column("job_id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("kind", sa.String(48), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("gpu", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("session_id", UUID(as_uuid=True), nullable=True),
        sa.Column("params", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("result", JSONB, nullable=True),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("stage", sa.String(64), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("app_user.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_tool_job_status", "tool_job", ["status", "gpu", "created_at"])
    op.create_index("ix_tool_job_kind", "tool_job", ["kind", "created_at"])
    op.create_index("ix_tool_job_session", "tool_job", ["session_id"])
    op.create_index("ix_tool_job_created_by", "tool_job", ["created_by"])


def downgrade() -> None:
    op.drop_index("ix_tool_job_created_by", table_name="tool_job")
    op.drop_index("ix_tool_job_session", table_name="tool_job")
    op.drop_index("ix_tool_job_kind", table_name="tool_job")
    op.drop_index("ix_tool_job_status", table_name="tool_job")
    op.drop_table("tool_job")
