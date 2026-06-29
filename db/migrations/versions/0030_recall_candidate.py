"""recall candidates: the recall-recovery audit row linking a recovered (review-state) Object to the
channels that proposed it and the human verdict (status), so the verdict can recalibrate each channel.

Revision ID: 0030_recall_candidate
Revises: 0029_lidar_provenance_gaps
Create Date: 2026-06-29
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY as PGARRAY
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0030_recall_candidate"
down_revision: str | None = "0029_lidar_provenance_gaps"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "recall_candidate",
        sa.Column("candidate_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("object_id", UUID(as_uuid=True),
                  sa.ForeignKey("object.object_id", ondelete="CASCADE"), nullable=False),
        sa.Column("frame_id", UUID(as_uuid=True),
                  sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False),
        sa.Column("channels", PGARRAY(sa.String(16)), nullable=True),
        sa.Column("fn_value", sa.Float(), nullable=False, server_default="0"),
        sa.Column("class_id", sa.Integer(), sa.ForeignKey("ontology_class.id"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_recall_candidate_status", "recall_candidate", ["status"])
    op.create_index("ix_recall_candidate_frame", "recall_candidate", ["frame_id"])

    # Widen the object-source CHECK (0020_constraints) to admit the recovery source.
    op.drop_constraint("ck_object_source", "object", type_="check")
    op.create_check_constraint(
        "ck_object_source", "object",
        "source IN ('fused', 'auto_accept', 'human', 'imported', 'relabel', 'interpolated', "
        "'propagated', 'recall')")


def downgrade() -> None:
    op.drop_constraint("ck_object_source", "object", type_="check")
    op.create_check_constraint(
        "ck_object_source", "object",
        "source IN ('fused', 'auto_accept', 'human', 'imported', 'relabel', 'interpolated', 'propagated')")
    op.drop_index("ix_recall_candidate_frame", table_name="recall_candidate")
    op.drop_index("ix_recall_candidate_status", table_name="recall_candidate")
    op.drop_table("recall_candidate")
