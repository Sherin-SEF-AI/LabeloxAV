"""Release immutability: store the content fingerprint on each dataset commit.

seal_commit_id hashes only object IDs, so a mutated bbox on the same object id would not change the commit id
(the release would silently drift). content_fingerprint hashes the annotation CONTENT (class, geometry, state).
Persisting it lets /release/{id}/verify recompute it from the live objects and detect any post-seal mutation.
Additive, nullable (old commits keep NULL and are reported as "not fingerprinted").

Revision ID: 0061_release_content_fp
Revises: 0060_champion_unique
Create Date: 2026-07-20
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0061_release_content_fp"
down_revision: str | None = "0060_champion_unique"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("dataset_commit", sa.Column("content_fingerprint", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("dataset_commit", "content_fingerprint")
