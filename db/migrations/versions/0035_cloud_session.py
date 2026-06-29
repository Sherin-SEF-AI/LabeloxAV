"""cloud_session: a warm cloud-GPU session (one RunPod pod held across a work session, torn down on
disconnect). The row drives the cost meter, the idle and max-session auto-terminate guards, and orphan
detection on app load, so a connected GPU can never silently run.

Revision ID: 0035_cloud_session
Revises: 0034_frame_segmentation
Create Date: 2026-06-29
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0035_cloud_session"
down_revision: str | None = "0034_frame_segmentation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cloud_session",
        sa.Column("session_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("pod_id", sa.Text(), nullable=True),
        sa.Column("mode", sa.String(8), nullable=False, server_default="warm"),
        sa.Column("state", sa.String(16), nullable=False, server_default="provisioning"),
        sa.Column("gpu_type", sa.String(64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("idle_since", sa.DateTime(timezone=True), nullable=True),
        sa.Column("gpu_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("est_cost", sa.Float(), nullable=False, server_default="0"),
        sa.Column("max_session_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_job_id", UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_cloud_session_state", "cloud_session", ["state"])


def downgrade() -> None:
    op.drop_index("ix_cloud_session_state", table_name="cloud_session")
    op.drop_table("cloud_session")
