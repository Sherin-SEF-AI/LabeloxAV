"""A verdict belongs to the batch it was produced for, not just to the object and the model.

The uniqueness was (object_id, judge, model_version), on the reasoning that one model asked the same
question about the same crop should give one answer. True, and it misses that the same object can appear in
two different populations being measured for two different reasons.

Judging a sample of error-detector candidates hit objects that were already in the calibration set. The
upsert fired, the row's batch_id was rewritten to the detector batch, and the calibration silently lost nine
of its 247 verdicts: not corrupted, moved. Nothing errored, and the calibration would have carried on
reporting a sensitivity computed over a set that had quietly shrunk.

Uniqueness therefore includes batch_id. An object judged in two batches by the same model keeps a row in
each, which is what the readers already assume: judged_precision filters by batch_id, stored_calibration
filters by batch_id, and machine_detector_weights filters by a batch prefix. Every one of them was written
expecting per-batch attribution that the constraint did not provide.

batch_id becomes NOT NULL for the same reason. Postgres treats NULLs as distinct in a unique index, so a
nullable column in the key would let unbounded duplicate rows accumulate for any verdict written without a
batch, which is exactly the case a constraint is supposed to prevent. No row is null today.
"""

import sqlalchemy as sa
from alembic import op

revision = "0086_machine_verdict_per_batch"
down_revision = "0085_workforce"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("machine_verdict", "batch_id",
                    existing_type=sa.String(64), nullable=False, server_default="")
    op.drop_constraint("uq_machine_verdict_object_judge", "machine_verdict", type_="unique")
    op.create_unique_constraint(
        "uq_machine_verdict_object_judge_batch", "machine_verdict",
        ["object_id", "judge", "model_version", "batch_id"])


def downgrade() -> None:
    # Going back means collapsing to one verdict per (object, judge, model). Duplicates across batches would
    # violate the old constraint, so the newest row per key wins and the rest are removed. Destructive by
    # nature, which is why it is spelled out rather than left to fail halfway.
    op.execute("""
        DELETE FROM machine_verdict a
        USING machine_verdict b
        WHERE a.object_id = b.object_id
          AND a.judge = b.judge
          AND a.model_version = b.model_version
          AND a.ts_ns < b.ts_ns
    """)
    op.drop_constraint("uq_machine_verdict_object_judge_batch", "machine_verdict", type_="unique")
    op.create_unique_constraint(
        "uq_machine_verdict_object_judge", "machine_verdict",
        ["object_id", "judge", "model_version"])
    op.alter_column("machine_verdict", "batch_id",
                    existing_type=sa.String(64), nullable=True, server_default=None)
