"""A machine's judgement on an existing label, kept out of the human review plane.

253 of 570,379 objects carry a Review row. Precision is unmeasurable, the gate cannot be tuned, the error
detector has 298,529 candidates and one confirmed verdict, and every quality claim the system makes rests on
labels nobody has checked. A VLM judge is the only thing that reaches that scale.

The obvious shortcut is to write the judge's opinion into `review` with reviewer='vlm'. That would be wrong
and hard to undo. `review` means a person ruled on this: `precision_batch` excludes reviewed objects from
sampling precisely because a human verdict is not evidence about unreviewed labels, `corpus_precision` reads
states a human moved, and the annotator scorecards count rows there. Machine opinions in that table would
corrupt all three, and the corruption would be invisible because the rows look identical.

So machine verdicts get their own table, and the two planes stay separable. That separation is not
bookkeeping: the whole method depends on being able to compare them. A judge has its own error rate, and the
only way to know it is to have humans rule on a subsample and measure the judge against them. Mixed into one
table there would be nothing to measure against.

Unique on (object_id, judge, model_version) so re-running the same judge is idempotent while a new model
version gets its own row, which is what makes two judges comparable on the same crops.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0082_machine_verdict"
down_revision = "0081_object_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "machine_verdict",
        sa.Column("verdict_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("object_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("object.object_id", ondelete="CASCADE"), nullable=False),
        # What kind of judge this was ("vlm"), separate from which model served it, so a future judge of a
        # different kind (an ensemble, a second detector) does not have to pretend to be a VLM.
        sa.Column("judge", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        # correct | incorrect | unsure. `unsure` is a real answer and is counted separately rather than
        # folded into either side, because a judge that abstains on the hard cases and is scored only on the
        # easy ones reports a precision that flatters itself.
        sa.Column("verdict", sa.String(16), nullable=False),
        sa.Column("proposed_class_id", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("agreement", sa.Float(), nullable=True),
        sa.Column("detail", postgresql.JSONB(astext_type=sa.Text()),
                  server_default=sa.text("'{}'::jsonb"), nullable=False),
        # The measurement batch this was produced for, so a judged sample stays identifiable as a sample.
        sa.Column("batch_id", sa.String(64), nullable=True),
        sa.Column("ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("verdict in ('correct','incorrect','unsure')", name="ck_machine_verdict_verdict"),
        sa.UniqueConstraint("object_id", "judge", "model_version", name="uq_machine_verdict_object_judge"),
    )
    op.create_index("ix_machine_verdict_batch", "machine_verdict", ["batch_id"])
    op.create_index("ix_machine_verdict_object", "machine_verdict", ["object_id"])


def downgrade() -> None:
    op.drop_index("ix_machine_verdict_object", table_name="machine_verdict")
    op.drop_index("ix_machine_verdict_batch", table_name="machine_verdict")
    op.drop_table("machine_verdict")
