"""Per-user token revocation: app_user.token_version.

Every v2 token carries this version; the verifier rejects a token whose version does not match the row.
Incrementing it revokes every token for that user at once, with no session store and without rotating the
global signing key. Existing rows default to 1, which matches tokens minted at version 1.

Revision ID: 0072_user_token_version
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0072_user_token_version"
down_revision = "0071_prediction_conf_nullable"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("app_user", sa.Column("token_version", sa.Integer(), nullable=False, server_default="1"))


def downgrade() -> None:
    op.drop_column("app_user", "token_version")
