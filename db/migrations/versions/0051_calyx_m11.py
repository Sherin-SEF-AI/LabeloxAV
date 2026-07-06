"""CALYX M11: data recovery. Per-session calibration confidence and a versioned calibration-override table so
a mildly drifted session is corrected and made usable instead of quarantined. Additive only.

Revision ID: 0051_calyx_m11
Revises: 0050_sanyx_m10
Create Date: 2026-07-06
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0051_calyx_m11"
down_revision: str | None = "0050_sanyx_m10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("calibration_validation", sa.Column("confidence", sa.Float(), nullable=True))
    op.create_table(
        "calibration_override",
        sa.Column("override_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("cam_id", sa.String(32), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("corrected", JSONB(), nullable=False, server_default="{}"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("provenance", JSONB(), nullable=False, server_default="{}"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("applied", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_calibration_override_session", "calibration_override", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_calibration_override_session", table_name="calibration_override")
    op.drop_table("calibration_override")
    op.drop_column("calibration_validation", "confidence")
