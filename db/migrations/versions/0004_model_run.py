"""model_run: fine-tune + eval-gate provenance (P1 close-the-loop)

Revision ID: 0004_model_run
Revises: 0003_embedding
Create Date: 2026-06-25
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_model_run"
down_revision: Union[str, None] = "0003_embedding"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_run",
        sa.Column("run_id", sa.String(128), primary_key=True),
        sa.Column("base_weights", sa.String(128), nullable=False),
        sa.Column("weights_uri", sa.Text()),
        sa.Column("dataset_name", sa.String(128), nullable=False),
        sa.Column("n_train", sa.Integer(), server_default="0"),
        sa.Column("n_val", sa.Integer(), server_default="0"),
        sa.Column("epochs", sa.Integer(), server_default="0"),
        sa.Column("metrics", postgresql.JSONB(), server_default="{}"),
        sa.Column("baseline_metrics", postgresql.JSONB(), server_default="{}"),
        sa.Column("gate", postgresql.JSONB(), server_default="{}"),
        sa.Column("promoted", sa.Boolean(), server_default=sa.false()),
        sa.Column("ontology_version", sa.String(64), nullable=False),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("model_run")
