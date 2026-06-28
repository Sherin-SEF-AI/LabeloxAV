"""Phase 2 Perception Depth schema: track id-switch/version, object temporal+recognition columns,
lane table, drivable_mask table

Revision ID: 0016_perception
Revises: 0015_intelligence
Create Date: 2026-06-27
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0016_perception"
down_revision: str | None = "0015_intelligence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # track: BoT-SORT outputs (M2.0)
    op.add_column("track", sa.Column("id_switch_flags", postgresql.JSONB()))
    op.add_column("track", sa.Column("tracker_version", sa.String(48)))

    # object: temporal (M2.5) + recognition (M2.3/M2.4)
    op.add_column("object", sa.Column("is_keyframe", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("object", sa.Column("interp_source", sa.String(16)))
    op.add_column("object", sa.Column("sign_type", sa.Text()))
    op.add_column("object", sa.Column("sign_category", sa.String(16)))
    op.add_column("object", sa.Column("ocr_text", sa.Text()))
    op.add_column("object", sa.Column("ocr_lang", sa.String(16)))
    op.add_column("object", sa.Column("ocr_conf", sa.Float()))

    # lane (M2.1)
    op.create_table(
        "lane",
        sa.Column("lane_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("track_ref", postgresql.UUID(as_uuid=True)),
        sa.Column("control_points", postgresql.JSONB(), nullable=False),
        sa.Column("lane_type", sa.String(16), nullable=False),
        sa.Column("is_ego", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("source", sa.String(16), nullable=False, server_default="proposed"),
        sa.Column("model_version", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_lane_frame", "lane", ["frame_id"])
    op.create_index("ix_lane_track_ref", "lane", ["track_ref"])

    # drivable_mask (M2.2)
    op.create_table(
        "drivable_mask",
        sa.Column("frame_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("mask_uri", sa.Text(), nullable=False),
        sa.Column("coverage", postgresql.JSONB(), server_default="{}"),
        sa.Column("source", sa.String(16), nullable=False, server_default="proposed"),
        sa.Column("model_version", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("drivable_mask")
    op.drop_table("lane")
    for col in ("ocr_conf", "ocr_lang", "ocr_text", "sign_category", "sign_type", "interp_source", "is_keyframe"):
        op.drop_column("object", col)
    op.drop_column("track", "tracker_version")
    op.drop_column("track", "id_switch_flags")
