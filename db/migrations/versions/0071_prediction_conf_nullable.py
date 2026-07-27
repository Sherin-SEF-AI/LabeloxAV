"""Make prediction.conf nullable so a reconstructed run can carry a null (unavailable) score.

A prediction backfilled from review history has no original confidence (Review.before never captured it), so
its conf is null and the eval refuses to compute AP/PR for it. A real inference run always writes a score.

Revision ID: 0071_prediction_conf_nullable
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0071_prediction_conf_nullable"
down_revision = "0070_eval_patch_run"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("prediction", "conf", existing_type=sa.Float(), nullable=True)


def downgrade() -> None:
    # Restore NOT NULL, defaulting any reconstructed (null-conf) rows to 0.0 so the constraint can re-apply.
    op.execute("UPDATE prediction SET conf = 0.0 WHERE conf IS NULL")
    op.alter_column("prediction", "conf", existing_type=sa.Float(), nullable=False)
