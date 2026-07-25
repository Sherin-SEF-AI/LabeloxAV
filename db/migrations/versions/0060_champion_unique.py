"""Promotion TOCTOU guard: at most one champion per task, enforced at the DB level.

Two concurrent promotions could each read "no better champion" and both call set_champion, leaving two rows
with is_champion=true for the same task (a split-brain serving decision). A partial unique index makes that
state impossible: the second committer fails instead of silently creating a second champion. Before creating
it, collapse any pre-existing duplicates to the most recently created champion per task.

Revision ID: 0060_champion_unique
Revises: 0059_hardening_m19
Create Date: 2026-07-20
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0060_champion_unique"
down_revision: str | None = "0059_hardening_m19"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Demote all but the newest champion per task, so the unique index below can be created cleanly.
    op.execute(
        """
        UPDATE model_registry m SET is_champion = false
        WHERE m.is_champion = true
          AND m.model_version <> (
              SELECT k.model_version FROM model_registry k
              WHERE k.task = m.task AND k.is_champion = true
              ORDER BY k.created_at DESC, k.model_version DESC
              LIMIT 1
          )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX uq_model_registry_champion "
        "ON model_registry (task) WHERE is_champion"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_model_registry_champion")
