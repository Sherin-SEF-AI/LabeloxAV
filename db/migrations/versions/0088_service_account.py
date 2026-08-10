"""Machine credentials, so an integration does not have to hold a person's password.

Every authentication path in this system was human: login, MFA, password reset. There was no way for a CI
job, a workforce provider's callback, a metrics exporter or a sibling application to call this API except by
being issued a person's twelve-hour bearer token, which also meant every action a machine took was recorded
as that person doing it.

A service account is deliberately still a user. It owns an `app_user` row and acts as it, so the existing
role floors, the audit trail and every `created_by` column keep working unchanged, and a machine's writes are
attributable exactly the way a person's are.

Two choices worth stating.

The key is never stored. The row holds the sha256 of the secret half and the public prefix that the lookup
indexes on, so a database someone walks off with yields no usable credential.

Revocation is a timestamp, not a version. A bearer token is checked against `token_version` and is otherwise
good until it expires, which is the wrong trade for a credential that will sit in a CI config for a year: it
has to stop working the moment somebody presses the button.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0088_service_account"
down_revision = "0087_agent_run_resumable"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "service_account",
        sa.Column("service_account_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(64), nullable=False, unique=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("app_user.user_id", ondelete="CASCADE"), nullable=False),
        sa.Column("key_prefix", sa.String(24), nullable=False, unique=True),
        sa.Column("key_hash", sa.String(64), nullable=False),
        sa.Column("scopes", postgresql.JSONB(astext_type=sa.Text()), server_default="[]", nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("created_by", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("app_user.user_id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"),
                  nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
    )
    # Every authenticated request with an API key resolves through this, so it is on the hot path.
    op.create_index("ix_service_account_key_prefix", "service_account", ["key_prefix"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_service_account_key_prefix", table_name="service_account")
    op.drop_table("service_account")
