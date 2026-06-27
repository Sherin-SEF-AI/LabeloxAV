"""P3 per-object dynamics: object_dynamics (distance/speed/heading/closing/ttc/risk derived from
track + ego speed + IPM ground-plane)

Revision ID: 0019_dynamics
Revises: 0018_closedloop
Create Date: 2026-06-27
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0019_dynamics"
down_revision: Union[str, None] = "0018_closedloop"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "object_dynamics",
        sa.Column("object_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("object.object_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("track_id", postgresql.UUID(as_uuid=True)),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True)),
        sa.Column("ts_ns", sa.BigInteger()),
        sa.Column("distance_m", sa.Float()),
        sa.Column("lateral_m", sa.Float()),
        sa.Column("speed_kmh", sa.Float()),
        sa.Column("closing_speed_kmh", sa.Float()),
        sa.Column("heading_deg", sa.Float()),
        sa.Column("ttc_s", sa.Float()),
        sa.Column("risk_level", sa.String(8)),
        sa.Column("method", sa.String(32), server_default="ipm_mono_v1"),
        sa.Column("confidence", sa.Float(), server_default="0.5"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_object_dynamics_track", "object_dynamics", ["track_id"])
    op.create_index("ix_object_dynamics_frame", "object_dynamics", ["frame_id"])


def downgrade() -> None:
    op.drop_table("object_dynamics")
