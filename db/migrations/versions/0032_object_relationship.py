"""object_relationship: a directed relationship between two objects on a frame, for grouping that
track_id cannot express (rider on a two-wheeler, trailer to truck, parent-child, herd membership).

Revision ID: 0032_object_relationship
Revises: 0031_object_polyline
Create Date: 2026-06-29
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0032_object_relationship"
down_revision: str | None = "0031_object_polyline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "object_relationship",
        sa.Column("relationship_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("from_object_id", UUID(as_uuid=True),
                  sa.ForeignKey("object.object_id", ondelete="CASCADE"), nullable=False),
        sa.Column("to_object_id", UUID(as_uuid=True),
                  sa.ForeignKey("object.object_id", ondelete="CASCADE"), nullable=False),
        sa.Column("frame_id", UUID(as_uuid=True),
                  sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(24), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_object_relationship_from", "object_relationship", ["from_object_id"])
    op.create_index("ix_object_relationship_to", "object_relationship", ["to_object_id"])
    op.create_index("ix_object_relationship_frame", "object_relationship", ["frame_id"])


def downgrade() -> None:
    for ix in ("ix_object_relationship_frame", "ix_object_relationship_to", "ix_object_relationship_from"):
        op.drop_index(ix, table_name="object_relationship")
    op.drop_table("object_relationship")
