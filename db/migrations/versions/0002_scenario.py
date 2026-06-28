"""scenario index (P2 scenario mining)

Revision ID: 0002_scenario
Revises: 0001_initial
Create Date: 2026-06-24
"""
from collections.abc import Sequence

import geoalchemy2
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_scenario"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "scenario",
        sa.Column("scenario_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE")),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("t_in_ns", sa.BigInteger(), nullable=False),
        sa.Column("t_out_ns", sa.BigInteger(), nullable=False),
        sa.Column("actors", postgresql.JSONB(), server_default="[]"),
        sa.Column("criticality", sa.Float(), server_default="0"),
        sa.Column("geo", geoalchemy2.Geography(geometry_type="POINT", srid=4326)),
        sa.Column("tags", postgresql.JSONB(), server_default="[]"),
        sa.Column("clip_ref", sa.Text()),
        sa.Column("meta", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_scenario_session", "scenario", ["session_id"])
    op.create_index("ix_scenario_type", "scenario", ["type"])
    op.create_index("ix_scenario_criticality", "scenario", ["criticality"])


def downgrade() -> None:
    op.drop_index("ix_scenario_criticality", table_name="scenario")
    op.drop_index("ix_scenario_type", table_name="scenario")
    op.drop_index("ix_scenario_session", table_name="scenario")
    op.drop_table("scenario")
