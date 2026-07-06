"""SIEVYX M12: directed long-tail discovery. Clip-level maneuver labels and auto-discovered scenario clusters.
Additive only.

Revision ID: 0052_sievyx_m12
Revises: 0051_calyx_m11
Create Date: 2026-07-06
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0052_sievyx_m12"
down_revision: str | None = "0051_calyx_m11"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "clip_maneuver",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("track_id", UUID(as_uuid=True), nullable=True),
        sa.Column("t_in_ns", sa.BigInteger(), nullable=False),
        sa.Column("t_out_ns", sa.BigInteger(), nullable=False),
        sa.Column("maneuver", sa.String(32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("features", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_clip_maneuver_session", "clip_maneuver", ["session_id"])
    op.create_table(
        "scenario_cluster",
        sa.Column("cluster_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("method", sa.String(24), nullable=False, server_default="dino_hdbscan"),
        sa.Column("size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rarity", sa.Float(), nullable=False, server_default="0"),
        sa.Column("rep_frame_ids", JSONB(), nullable=False, server_default="[]"),
        sa.Column("name", sa.String(64), nullable=True),
        sa.Column("status", sa.String(12), nullable=False, server_default="discovered"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("scenario_cluster")
    op.drop_index("ix_clip_maneuver_session", table_name="clip_maneuver")
    op.drop_table("clip_maneuver")
