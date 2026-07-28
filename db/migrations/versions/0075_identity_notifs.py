"""Identity, notifications, the PII access log, the activity feed, and sealed track ids.

Five gaps that all took the same shape: the system knew something and had nowhere to put it.

- There was no way for a person to obtain access themselves. Every credential was admin-minted, with no
  password, no second factor, and no route to a corporate directory. That was the single largest blocker to
  a real deployment.
- Issue comments, job completions, gate blocks, drift breaches, and SLO alarms were all silent.
- `pii_audit` recorded what the redactor found and nothing recorded what a human then looked at, which is
  the half of the question a DPDPA enquiry actually asks.
- Reviews, objects, and jobs each kept their own history and none of them was a timeline.
- Tracking metrics existed and could never be scored, because a sealed gold set carried no track ids.

Revision ID: 0075_identity_notifs
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0075_identity_notifs"
down_revision = "0074_labeloxsec_anpr"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- identity -------------------------------------------------------------------------------
    op.create_table(
        "user_credential",
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("app_user.user_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("password_hash", sa.Text()),
        sa.Column("password_set_at", sa.DateTime(timezone=True)),
        sa.Column("failed_attempts", sa.Integer(), nullable=False, server_default="0"),
        # In the row, not in a cache: a lockout a process restart clears is not a lockout.
        sa.Column("locked_until", sa.DateTime(timezone=True)),
        sa.Column("totp_secret", sa.Text()),
        sa.Column("totp_confirmed_at", sa.DateTime(timezone=True)),
        sa.Column("recovery_hashes", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("last_totp_step", sa.BigInteger()),
        sa.Column("oidc_issuer", sa.Text()),
        sa.Column("oidc_subject", sa.Text()),
        sa.Column("email", sa.String(320)),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    # Unique on issuer+subject, never on email: a directory can reassign an address to a different person,
    # and matching on it would hand them the previous holder's account.
    op.create_index("ix_user_credential_oidc", "user_credential", ["oidc_issuer", "oidc_subject"],
                    unique=True)
    op.create_index("ix_user_credential_email", "user_credential", ["email"])

    op.create_table(
        "password_reset",
        sa.Column("reset_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("app_user.user_id", ondelete="CASCADE")),
        # The hash, not the token: a database read must not be convertible into an account takeover.
        sa.Column("token_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_password_reset_user", "password_reset", ["user_id"])
    op.create_index("ix_password_reset_token", "password_reset", ["token_hash"])

    # ---- notifications --------------------------------------------------------------------------
    op.create_table(
        "notification",
        sa.Column("notification_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("app_user.user_id", ondelete="CASCADE")),
        sa.Column("role", sa.String(16)),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False, server_default="info"),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("body", sa.Text()),
        sa.Column("href", sa.Text()),
        sa.Column("subject_type", sa.String(32)),
        sa.Column("subject_id", sa.String(64)),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("read_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_notification_user", "notification", ["user_id"])
    op.create_index("ix_notification_role", "notification", ["role"])
    op.create_index("ix_notification_kind", "notification", ["kind"])
    op.create_index("ix_notification_created", "notification", ["created_at"])
    # The bell's own query: this user's unread, newest first.
    op.create_index("ix_notification_user_unread", "notification", ["user_id", "read_at"])
    # Supersede rather than pile up: a second event about the same subject finds the first through this.
    op.create_index("ix_notification_subject", "notification", ["subject_type", "subject_id"])

    op.create_table(
        "notification_read",
        sa.Column("notification_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("notification.notification_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("app_user.user_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("read_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ---- PII access log -------------------------------------------------------------------------
    op.create_table(
        "pii_access_log",
        sa.Column("access_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        # SET NULL, not CASCADE: the fact that access happened outlives the account that made it, and an
        # erasure enquiry needs to see it even after the actor is gone.
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("app_user.user_id", ondelete="SET NULL")),
        sa.Column("user_name", sa.String(64)),
        sa.Column("subject_type", sa.String(32), nullable=False),
        sa.Column("subject_id", sa.String(64), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("session.session_id", ondelete="SET NULL")),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("pii_kinds", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("redacted", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("route", sa.Text()),
        sa.Column("pack_id", sa.String(32)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_pii_access_user", "pii_access_log", ["user_id"])
    op.create_index("ix_pii_access_subject", "pii_access_log", ["subject_id"])
    op.create_index("ix_pii_access_session", "pii_access_log", ["session_id"])
    op.create_index("ix_pii_access_created", "pii_access_log", ["created_at"])
    op.create_index("ix_pii_access_user_time", "pii_access_log", ["user_id", "created_at"])

    # ---- activity feed --------------------------------------------------------------------------
    op.create_table(
        "activity_event",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("app_user.user_id", ondelete="CASCADE")),
        sa.Column("user_name", sa.String(64)),
        sa.Column("verb", sa.String(32), nullable=False),
        sa.Column("subject_type", sa.String(32)),
        sa.Column("subject_id", sa.String(64)),
        sa.Column("summary", sa.Text()),
        sa.Column("href", sa.Text()),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_activity_user", "activity_event", ["user_id"])
    op.create_index("ix_activity_verb", "activity_event", ["verb"])
    op.create_index("ix_activity_created", "activity_event", ["created_at"])
    op.create_index("ix_activity_user_time", "activity_event", ["user_id", "created_at"])

    # ---- sealed track ids -----------------------------------------------------------------------
    # Defaults are empty and false, so every gold set sealed before this migration correctly reports that it
    # cannot support a tracking evaluation, rather than appearing to and scoring against nothing.
    op.add_column("gold_set", sa.Column("track_ids", postgresql.JSONB(), nullable=False,
                                        server_default="[]"))
    op.add_column("gold_set", sa.Column("tracks_sealed", sa.Boolean(), nullable=False,
                                        server_default=sa.false()))


def downgrade() -> None:
    op.drop_column("gold_set", "tracks_sealed")
    op.drop_column("gold_set", "track_ids")
    op.drop_table("activity_event")
    op.drop_table("pii_access_log")
    op.drop_table("notification_read")
    op.drop_table("notification")
    op.drop_table("password_reset")
    op.drop_table("user_credential")
