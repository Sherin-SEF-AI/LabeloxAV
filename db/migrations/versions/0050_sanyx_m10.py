"""SANYX M10: named root cause + operator remediation on the health report, and a rig-level predictive
maintenance alert table. Additive only.

Revision ID: 0050_sanyx_m10
Revises: 0049_data_engine_spine
Create Date: 2026-07-06
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0050_sanyx_m10"
down_revision: str | None = "0049_data_engine_spine"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("session_health", sa.Column("root_cause", sa.String(48), nullable=True))
    op.add_column("session_health", sa.Column("remediation", sa.Text(), nullable=True))
    op.create_table(
        "sanyx_rig_alert",
        sa.Column("alert_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("vehicle_id", sa.String(64), nullable=False),
        sa.Column("component", sa.String(32), nullable=False),
        sa.Column("metric", sa.String(48), nullable=False),
        sa.Column("trend", sa.String(8), nullable=False),
        sa.Column("severity", sa.String(8), nullable=False),
        sa.Column("evidence", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_sanyx_rig_alert_vehicle", "sanyx_rig_alert", ["vehicle_id"])


def downgrade() -> None:
    op.drop_index("ix_sanyx_rig_alert_vehicle", table_name="sanyx_rig_alert")
    op.drop_table("sanyx_rig_alert")
    op.drop_column("session_health", "remediation")
    op.drop_column("session_health", "root_cause")
