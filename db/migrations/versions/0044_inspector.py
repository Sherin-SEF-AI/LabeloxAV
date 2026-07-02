"""Session Inspector tables: session_index (M-I.1 MCAP index), session_health (M-I.2 health verdicts),
inspector_layout (M-I.3 saveable panel layouts).

Revision ID: 0044_inspector
Revises: 0043_collection_order
Create Date: 2026-07-02
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0044_inspector"
down_revision: str | None = "0043_collection_order"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "session_index",
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id"), primary_key=True),
        sa.Column("mcap_uri", sa.Text(), nullable=True),
        sa.Column("topics", JSONB(), nullable=False, server_default="{}"),
        sa.Column("time_range", sa.ARRAY(sa.BigInteger()), nullable=True),
        sa.Column("gaps", JSONB(), nullable=False, server_default="{}"),
        sa.Column("indexer_version", sa.String(32), nullable=False),
        sa.Column("built_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "session_health",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id"), nullable=False),
        sa.Column("checks", JSONB(), nullable=False, server_default="[]"),
        sa.Column("verdict", sa.String(8), nullable=False),
        sa.Column("indexer_version", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_session_health_session", "session_health", ["session_id"])
    op.create_table(
        "inspector_layout",
        sa.Column("layout_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("app_user.user_id"), nullable=True),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("panels", JSONB(), nullable=False, server_default="[]"),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("inspector_layout")
    op.drop_index("ix_session_health_session", table_name="session_health")
    op.drop_table("session_health")
    op.drop_table("session_index")
