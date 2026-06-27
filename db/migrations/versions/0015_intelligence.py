"""Data Intelligence Layer schema: pgvector frame/object embeddings, frame scene/dup/selected columns,
scenario_candidate discovery queue

Revision ID: 0015_intelligence
Revises: 0014_pgvector
Create Date: 2026-06-27
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0015_intelligence"
down_revision: Union[str, None] = "0014_pgvector"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Replace the legacy DINOv2 ARRAY frame_embedding with the pgvector DINOv3 + SigLIP 2 schema.
    op.drop_table("frame_embedding")
    op.create_table(
        "frame_embedding",
        sa.Column("frame_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("dino_vec", Vector(768)),
        sa.Column("siglip_vec", Vector(1152)),
        sa.Column("model_versions", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_frame_emb_dino_hnsw", "frame_embedding", ["dino_vec"],
                    postgresql_using="hnsw", postgresql_ops={"dino_vec": "vector_cosine_ops"})
    op.create_index("ix_frame_emb_siglip_hnsw", "frame_embedding", ["siglip_vec"],
                    postgresql_using="hnsw", postgresql_ops={"siglip_vec": "vector_cosine_ops"})

    op.create_table(
        "object_embedding",
        sa.Column("object_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("object.object_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("dino_vec", Vector(768), nullable=False),
        sa.Column("model_versions", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_object_emb_dino_hnsw", "object_embedding", ["dino_vec"],
                    postgresql_using="hnsw", postgresql_ops={"dino_vec": "vector_cosine_ops"})

    # Frame intelligence columns (nullable + additive).
    op.add_column("frame", sa.Column("scene", postgresql.JSONB()))
    op.add_column("frame", sa.Column("dup_group_id", postgresql.UUID(as_uuid=True)))
    op.add_column("frame", sa.Column("is_dup_canonical", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("frame", sa.Column("dup_score", sa.Float()))
    op.add_column("frame", sa.Column("selected", sa.Boolean(), nullable=False, server_default=sa.true()))
    op.add_column("frame", sa.Column("novelty_score", sa.Float()))
    op.create_index("ix_frame_dup_group", "frame", ["dup_group_id"])
    op.create_index("ix_frame_selected", "frame", ["session_id", "selected"])

    op.create_table(
        "scenario_candidate",
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("frame.frame_id", ondelete="CASCADE")),
        sa.Column("kind", sa.String(24), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("cluster_id", sa.Integer()),
        sa.Column("rare_classes", postgresql.ARRAY(sa.Text())),
        sa.Column("state", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("tag", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_scenario_candidate_state", "scenario_candidate", ["state", "score"])


def downgrade() -> None:
    op.drop_table("scenario_candidate")
    for col in ("novelty_score", "selected", "dup_score", "is_dup_canonical", "dup_group_id", "scene"):
        op.drop_column("frame", col)
    op.drop_table("object_embedding")
    op.drop_table("frame_embedding")
    op.create_table(
        "frame_embedding",
        sa.Column("frame_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("model", sa.String(48), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("vec", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
