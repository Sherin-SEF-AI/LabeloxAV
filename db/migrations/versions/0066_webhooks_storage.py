"""Webhooks and registered storage sources.

Webhook deliveries are HMAC-signed with a per-subscription secret. Without a signature a receiver cannot
distinguish a real delivery from anyone who learned the URL, which turns a webhook into an unauthenticated
write into whatever it triggers.

storage_source deliberately stores no credentials, only the locator and the name of a server-side credential
profile. Keeping per-source cloud keys in the application database would make one read of one table a breach
of every connected bucket.

Additive only.

Revision ID: 0066_webhooks_storage
Revises: 0065_asset_annotation
Create Date: 2026-07-24
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0066_webhooks_storage"
down_revision: str | None = "0065_asset_annotation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UUID = postgresql.UUID(as_uuid=True)
_JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "webhook",
        sa.Column("webhook_id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("project_id", _UUID, sa.ForeignKey("label_project.project_id", ondelete="CASCADE"),
                  nullable=True),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("events", _JSONB, server_default="[]"),
        sa.Column("secret", sa.String(length=128), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("last_status", sa.Integer(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_delivery_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_webhook_project", "webhook", ["project_id", "active"])

    op.create_table(
        "storage_source",
        sa.Column("source_id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("project_id", _UUID, sa.ForeignKey("label_project.project_id", ondelete="CASCADE"),
                  nullable=True),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("provider", sa.String(length=12), nullable=False),
        sa.Column("bucket", sa.String(length=200), nullable=False),
        sa.Column("prefix", sa.Text(), nullable=True),
        sa.Column("region", sa.String(length=32), nullable=True),
        sa.Column("endpoint_url", sa.Text(), nullable=True),
        sa.Column("credential_profile", sa.String(length=64), nullable=True),
        sa.Column("last_scanned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_object_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_storage_source_project", "storage_source", ["project_id"])


def downgrade() -> None:
    op.drop_table("storage_source")
    op.drop_table("webhook")
