"""R3 optimistic concurrency: object.version, bumped on every human edit so a stale write is rejected
(409) instead of silently overwriting another annotator's change.

Revision ID: 0021_object_version
Revises: 0020_constraints
Create Date: 2026-06-28
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0021_object_version"
down_revision: str | None = "0020_constraints"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("object", sa.Column("version", sa.Integer(), nullable=False, server_default="1"))


def downgrade() -> None:
    op.drop_column("object", "version")
