"""Multi-camera annotation tables: frame_group (M-MC.0 synchronized rig frames), rig_object (M-MC.2 one
physical object across views), plus object.rig_object_id.

Revision ID: 0045_multicam
Revises: 0044_inspector
Create Date: 2026-07-02
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0045_multicam"
down_revision: str | None = "0044_inspector"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "frame_group",
        sa.Column("group_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id"), nullable=False),
        sa.Column("ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("frame_ids", JSONB(), nullable=False, server_default="{}"),
        sa.Column("missing_cams", sa.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("sync_spread_ns", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("n_cams", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("confirmed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_frame_group_session_ts", "frame_group", ["session_id", "ts_ns"])
    op.create_table(
        "rig_object",
        sa.Column("rig_object_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id"), nullable=False),
        sa.Column("group_id", UUID(as_uuid=True), sa.ForeignKey("frame_group.group_id"), nullable=False),
        sa.Column("class_id", sa.Integer(), nullable=True),
        sa.Column("member_object_ids", sa.ARRAY(UUID(as_uuid=True)), nullable=False, server_default="{}"),
        sa.Column("link_sources", JSONB(), nullable=False, server_default="{}"),
        sa.Column("rig_track_id", UUID(as_uuid=True), nullable=True),
        sa.Column("conflict", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("provenance", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_rig_object_group", "rig_object", ["group_id"])
    op.create_index("ix_rig_object_track", "rig_object", ["rig_track_id"])
    op.add_column("object", sa.Column("rig_object_id", UUID(as_uuid=True), nullable=True))


def downgrade() -> None:
    op.drop_column("object", "rig_object_id")
    op.drop_index("ix_rig_object_track", table_name="rig_object")
    op.drop_index("ix_rig_object_group", table_name="rig_object")
    op.drop_table("rig_object")
    op.drop_index("ix_frame_group_session_ts", table_name="frame_group")
    op.drop_table("frame_group")
