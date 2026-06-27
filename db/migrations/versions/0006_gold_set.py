"""gold_set: frozen, versioned gold set from the fleet's own frames (Gate B, M9)

Revision ID: 0006_gold_set
Revises: 0005_pii_audit
Create Date: 2026-06-26
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_gold_set"
down_revision: Union[str, None] = "0005_pii_audit"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "gold_set",
        sa.Column("gold_id", sa.String(128), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("spec", postgresql.JSONB(), server_default="{}"),
        sa.Column("object_ids", postgresql.JSONB(), server_default="[]"),
        sa.Column("n_objects", sa.Integer(), server_default="0"),
        sa.Column("n_frames", sa.Integer(), server_default="0"),
        sa.Column("ontology_version", sa.String(64), nullable=False),
        sa.Column("metrics", postgresql.JSONB(), server_default="{}"),
        sa.Column("data_yaml_uri", sa.Text()),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("gold_set")
