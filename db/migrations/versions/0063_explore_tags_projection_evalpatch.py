"""Explorer foundation: curation tags, persisted 2D embedding projections, and evaluation patches.

Three additive pieces behind the FiftyOne-style Explore workspace:
- `tags` on frame and object (JSONB + GIN), the free-form curation marks the explorer applies in bulk. These
  are separate from frame.scene (model-derived) and object.attrs (ontology-typed), which keep their meaning.
- embedding_projection / embedding_projection_point: a fitted 2D layout of an existing pgvector space, stored
  so the map opens instantly and reproduces exactly instead of being re-fit per visit.
- eval_patch: per-prediction tp/fp/fn outcomes against a sealed gold set, so a confusion cell can be opened
  and the actual crops inspected.

Additive only: no existing column changes, so it is safe on a populated corpus.

Revision ID: 0063_explore_foundation
Revises: 0062_cascade_and_scene_gin
Create Date: 2026-07-24
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0063_explore_foundation"
down_revision: str | None = "0062_cascade_and_scene_gin"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ---- curation tags on frame + object
    for table in ("frame", "object"):
        op.add_column(table, sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()),
                                       nullable=False, server_default="[]"))
        op.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_tags_gin ON {table} USING gin (tags)")

    # ---- fitted 2D embedding projections
    op.create_table(
        "embedding_projection",
        sa.Column("projection_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("kind", sa.String(length=8), nullable=False),
        sa.Column("space", sa.String(length=8), nullable=False),
        sa.Column("method", sa.String(length=8), nullable=False),
        sa.Column("params", postgresql.JSONB(astext_type=sa.Text()), server_default="{}"),
        sa.Column("n", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "embedding_projection_point",
        sa.Column("projection_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("embedding_projection.projection_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("ref_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("x", sa.Float(), nullable=False),
        sa.Column("y", sa.Float(), nullable=False),
        sa.Column("cluster", sa.Integer(), nullable=True),
    )
    op.create_index("ix_projection_point_projection", "embedding_projection_point", ["projection_id"])

    # ---- evaluation patches (confusion-cell drill-down)
    op.create_table(
        "eval_patch",
        sa.Column("patch_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("eval_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("gold_id", sa.String(length=128), nullable=True),
        sa.Column("model_version", sa.String(length=128), nullable=True),
        sa.Column("object_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("object.object_id", ondelete="CASCADE"), nullable=True),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=True),
        sa.Column("outcome", sa.String(length=4), nullable=False),
        sa.Column("gt_class_id", sa.Integer(), nullable=True),
        sa.Column("pred_class_id", sa.Integer(), nullable=True),
        sa.Column("iou", sa.Float(), nullable=True),
        sa.Column("conf", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_eval_patch_eval", "eval_patch", ["eval_id"])
    op.create_index("ix_eval_patch_cell", "eval_patch", ["eval_id", "gt_class_id", "pred_class_id"])


def downgrade() -> None:
    op.drop_table("eval_patch")
    op.drop_table("embedding_projection_point")
    op.drop_table("embedding_projection")
    for table in ("frame", "object"):
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_tags_gin")
        op.drop_column(table, "tags")
