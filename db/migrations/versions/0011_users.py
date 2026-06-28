"""app_user + review.user_id: lightweight multi-user (accounts, roles, attribution)

Revision ID: 0011_users
Revises: 0010_autolabel_job
Create Date: 2026-06-27
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_users"
down_revision: str | None = "0010_autolabel_job"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "app_user",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(64), nullable=False, unique=True),
        sa.Column("role", sa.String(16), nullable=False, server_default="annotator"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.add_column("review", sa.Column("user_id", postgresql.UUID(as_uuid=True),
                                      sa.ForeignKey("app_user.user_id", ondelete="SET NULL")))
    # seed a default admin so there is always a usable identity
    op.execute("INSERT INTO app_user (user_id, name, role) VALUES (gen_random_uuid(), 'admin', 'admin')")


def downgrade() -> None:
    op.drop_column("review", "user_id")
    op.drop_table("app_user")
