"""Two annotators' boxes on one frame were indistinguishable, so agreement could not be measured.

`object` records what a label says and never recorded who made it or which job it came from. `source` says
`human` and `provenance` is free-form, so two people working the same frame produced one undifferentiated
pile of rows. Every downstream measurement inherits that: `score_honeypots` grades "all human objects on the
frame", which with two annotators grades one against the other's hidden gold.

`job_id` and `annotator_id` are nullable, which is what makes this safe on 576,393 existing rows: nothing is
rewritten, nothing is backfilled wholesale, and every object that predates this keeps meaning exactly what
it meant. The one bounded exception is vendor work: `services/labelops/return_ingest.py` already writes the
job id into `provenance`, so those rows are promoted into the column here. That subset is small, it is the
only place in the tree that already knew its job, and leaving two sources of the same truth to drift would
be worse than moving it.

**The index is built CONCURRENTLY** because a plain CREATE INDEX takes ACCESS EXCLUSIVE on `object` for its
whole duration, and this table is read by the editor on every frame open.

`replica_group` is how two jobs over the same frames know they are a pair. Their `frame_ids` cannot be
compared to recover it: `seed_honeypots` appends gold frames chosen from `random.Random(job_id)`, so two
replicas diverge the moment they are created.
"""

import sqlalchemy as sa
from alembic import op

revision = "0091_job_attribution"
down_revision = "0090_drop_legacy_embedding"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("object", sa.Column("job_id", sa.UUID(), nullable=True))
    op.add_column("object", sa.Column("annotator_id", sa.UUID(), nullable=True))
    op.create_foreign_key("fk_object_job", "object", "label_job", ["job_id"], ["job_id"],
                          ondelete="SET NULL")
    op.create_foreign_key("fk_object_annotator", "object", "app_user", ["annotator_id"], ["user_id"],
                          ondelete="SET NULL")

    op.add_column("label_job", sa.Column("replica_group", sa.UUID(), nullable=True))
    op.add_column("label_job", sa.Column("replica_index", sa.Integer(), nullable=False,
                                         server_default="0"))
    op.add_column("label_task", sa.Column("replicas", sa.Integer(), nullable=False, server_default="1"))
    op.create_index("ix_label_job_replica_group", "label_job", ["replica_group"])

    # Vendor returns already carry their job in provenance. The EXISTS guard keeps a since-deleted job from
    # failing the new foreign key.
    op.execute("""
        UPDATE object SET job_id = (provenance ->> 'job_id')::uuid
        WHERE provenance ? 'job_id'
          AND EXISTS (SELECT 1 FROM label_job lj WHERE lj.job_id = (provenance ->> 'job_id')::uuid)
    """)

    op.create_table(
        "job_agreement",
        sa.Column("agreement_id", sa.UUID(), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("task_id", sa.UUID(), sa.ForeignKey("label_task.task_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("replica_group", sa.UUID(), nullable=False),
        sa.Column("frame_id", sa.UUID(), sa.ForeignKey("frame.frame_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("job_a_id", sa.UUID(), sa.ForeignKey("label_job.job_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("job_b_id", sa.UUID(), sa.ForeignKey("label_job.job_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("metrics", sa.dialects.postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("n_disagreements", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        # One verdict per frame per pair, so recomputing a group updates rather than accumulates.
        sa.UniqueConstraint("frame_id", "job_a_id", "job_b_id", name="uq_job_agreement_pair"),
    )
    op.create_index("ix_job_agreement_task", "job_agreement", ["task_id"])
    op.create_index("ix_job_agreement_group", "job_agreement", ["replica_group"])

    # Outside a transaction: Postgres refuses CREATE INDEX CONCURRENTLY inside one, and Alembic runs a
    # migration in a transaction by default.
    with op.get_context().autocommit_block():
        op.execute("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_object_job_frame
            ON object (job_id, frame_id) WHERE job_id IS NOT NULL
        """)


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_object_job_frame")

    op.drop_index("ix_job_agreement_group", table_name="job_agreement")
    op.drop_index("ix_job_agreement_task", table_name="job_agreement")
    op.drop_table("job_agreement")

    op.drop_index("ix_label_job_replica_group", table_name="label_job")
    op.drop_column("label_task", "replicas")
    op.drop_column("label_job", "replica_index")
    op.drop_column("label_job", "replica_group")

    op.drop_constraint("fk_object_annotator", "object", type_="foreignkey")
    op.drop_constraint("fk_object_job", "object", type_="foreignkey")
    op.drop_column("object", "annotator_id")
    op.drop_column("object", "job_id")
    # Dropping job_id loses nothing that was not also written to provenance: return_ingest keeps writing
    # both for exactly this reason, and in-house creates predate nothing (the Review row with action
    # "create" carries the same user).
