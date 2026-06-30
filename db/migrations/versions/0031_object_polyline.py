"""object.polyline: open polyline geometry (curb, road_edge, barrier) on the object, so a linear feature
is a first-class object that flows through the gate, review queue, and export like any other.

Revision ID: 0031_object_polyline
Revises: 0030_recall_candidate
Create Date: 2026-06-29
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0031_object_polyline"
down_revision: str | None = "0030_recall_candidate"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("object", sa.Column("polyline", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("object", "polyline")
