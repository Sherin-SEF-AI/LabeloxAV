"""training_job.compute_target: route a job to the local GPU or the cloud A100 (hybrid compute)

Revision ID: 0009_compute_target
Revises: 0008_training_job
Create Date: 2026-06-26
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0009_compute_target"
down_revision: Union[str, None] = "0008_training_job"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Metadata-only add with a server_default (instant in Postgres, safe while the worker runs a job).
    op.add_column("training_job", sa.Column("compute_target", sa.String(16), nullable=False, server_default="local"))


def downgrade() -> None:
    op.drop_column("training_job", "compute_target")
