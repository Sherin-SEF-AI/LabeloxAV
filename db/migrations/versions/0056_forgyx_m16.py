"""FORGYX M16: hardware-in-the-loop. Signed deployment packaging, thermal envelope, and rollout/rollback state
on deployment. Additive only.

Revision ID: 0056_forgyx_m16
Revises: 0055_verdyx_m15
Create Date: 2026-07-06
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0056_forgyx_m16"
down_revision: str | None = "0055_verdyx_m15"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("deployment", sa.Column("signature", sa.String(length=128), nullable=True))
    op.add_column("deployment", sa.Column("package_uri", sa.Text(), nullable=True))
    op.add_column("deployment", sa.Column("thermal_envelope", postgresql.JSONB(astext_type=sa.Text()),
                                          server_default=sa.text("'{}'::jsonb"), nullable=False))
    op.add_column("deployment", sa.Column("rollout_state", sa.String(length=12),
                                          server_default="none", nullable=False))
    op.add_column("deployment", sa.Column("superseded_by", postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key("fk_deployment_superseded_by", "deployment", "deployment",
                          ["superseded_by"], ["deployment_id"], ondelete="SET NULL")


def downgrade() -> None:
    op.drop_constraint("fk_deployment_superseded_by", "deployment", type_="foreignkey")
    op.drop_column("deployment", "superseded_by")
    op.drop_column("deployment", "rollout_state")
    op.drop_column("deployment", "thermal_envelope")
    op.drop_column("deployment", "package_uri")
    op.drop_column("deployment", "signature")
