"""collection_order: the Fleet Dispatch agent's daily proposal to send a vehicle to collect the data the
corpus is starved of.

Revision ID: 0043_collection_order
Revises: 0042_promotion_proposal
Create Date: 2026-07-02
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0043_collection_order"
down_revision: str | None = "0042_promotion_proposal"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "collection_order",
        sa.Column("order_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("vehicle_id", sa.String(64), nullable=False),
        sa.Column("city", sa.String(64), nullable=True),
        sa.Column("area", sa.String(128), nullable=True),
        sa.Column("window", sa.String(32), nullable=True),
        sa.Column("target", sa.Text(), nullable=False),
        sa.Column("gap_kind", sa.String(24), nullable=True),
        sa.Column("forecast", sa.String(32), nullable=True),
        sa.Column("priority", sa.Float(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(16), nullable=False, server_default="proposed"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("created_by", sa.String(64), nullable=True),
    )
    op.create_index("ix_collection_order_status", "collection_order", ["status"])


def downgrade() -> None:
    op.drop_index("ix_collection_order_status", table_name="collection_order")
    op.drop_table("collection_order")
