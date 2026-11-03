"""Index every foreign key that points at object.

Deleting an object makes Postgres find its rows in each referencing table, for the cascade or the SET NULL.
Without an index on the referencing column that is a full scan of the table, once per deleted object.
eval_patch holds 36 million rows, so a re-label pass, which clears a frame's machine objects before writing
new ones, spent minutes on a single frame and never finished its first drive overnight. The smaller tables
had the same gap and would have grown into it.

Built CONCURRENTLY so that writes to these tables, eval runs above all, are not blocked for the length of
the build.

Revision ID: 0122_object_fk_indexes
Revises: 0121_track_tube_score
"""

from alembic import op

revision = "0122_object_fk_indexes"
down_revision = "0121_track_tube_score"
branch_labels = None
depends_on = None

INDEXES = (
    ("ix_eval_patch_object", "eval_patch", "object_id"),
    ("ix_shadow_disagreement_verdict_object", "shadow_disagreement", "verdict_object_id"),
    ("ix_control_sample_object", "control_sample", "object_id"),
    ("ix_issue_object", "issue", "object_id"),
    ("ix_recall_candidate_object", "recall_candidate", "object_id"),
    ("ix_propagation_conflict_object", "propagation_conflict", "object_id"),
    ("ix_settlement_spot_object", "settlement_spot", "object_id"),
)


def upgrade() -> None:
    # Outside a transaction: Postgres refuses CREATE INDEX CONCURRENTLY inside one.
    with op.get_context().autocommit_block():
        for name, table, column in INDEXES:
            op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON {table} ({column})")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name, _table, _column in INDEXES:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
