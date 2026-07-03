"""M-F.5 scene-graph relations + VLM dataset generation: extend object_relationship with status/source/evidence
for proposed scene-graph relations, and add vlm_target for grounded multimodal training targets.

Revision ID: 0048_scenegraph_vlm
Revises: 0047_track_intent
Create Date: 2026-07-03
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0048_scenegraph_vlm"
down_revision: str | None = "0047_track_intent"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column("object_relationship", "kind", type_=sa.String(32))
    op.add_column("object_relationship", sa.Column("status", sa.String(16), nullable=False, server_default="confirmed"))
    op.add_column("object_relationship", sa.Column("source", sa.String(16), nullable=False, server_default="human"))
    op.add_column("object_relationship", sa.Column("evidence", JSONB(), nullable=True))
    op.create_table(
        "vlm_target",
        sa.Column("target_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("frame_id", UUID(as_uuid=True), sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(24), nullable=False),
        sa.Column("content", JSONB(), nullable=False, server_default="{}"),
        sa.Column("grounding", JSONB(), nullable=False, server_default="{}"),
        sa.Column("model", sa.String(48), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="generated"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_vlm_target_frame", "vlm_target", ["frame_id"])
    op.create_index("ix_vlm_target_status", "vlm_target", ["status"])


def downgrade() -> None:
    op.drop_table("vlm_target")
    op.drop_column("object_relationship", "evidence")
    op.drop_column("object_relationship", "source")
    op.drop_column("object_relationship", "status")
    op.alter_column("object_relationship", "kind", type_=sa.String(24))
