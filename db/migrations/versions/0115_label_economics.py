"""What a label costs and what it is worth, so a labelling budget can be spent rather than guessed.

The engine can already say which classes the gate is blocked on and roughly how many verdicts a
settlement lot needs. It cannot say what any of that costs, so every allocation decision has been made on
counts alone: a class needing 500 verdicts looks twice as expensive as one needing 250, whatever either
is actually worth or however long its crops take to judge.

Two additions.

Rates on `workforce`, because a verdict and a box are different work at different prices and neither is a
constant across a workforce. Nullable, and null means nobody has entered a rate: a default rate would put
a fabricated number into every value calculation and there would be no way to tell it from a real one.

`label_value_snapshot` records the ranking at a moment. A snapshot rather than a live view, because the
inputs move (the gate's deficit changes with every retrain, the measured minutes change with every
review) and a decision made last Tuesday has to be explainable against what was true last Tuesday.

`measured` is on the row for the same reason it is on every other table here: a class with too few timed
reviews to estimate minutes from gets a row saying so, not a row with an assumed median. The gate can
then tell "this class is cheap" from "nobody has timed this class".

Revision ID: 0115_label_economics
Revises: 0114_ontology_versions
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0115_label_economics"
down_revision = "0114_ontology_versions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("workforce", sa.Column("rate_inr_per_verdict", sa.Float(), nullable=True))
    op.add_column("workforce", sa.Column("rate_inr_per_box", sa.Float(), nullable=True))

    op.create_table(
        "label_value_snapshot",
        sa.Column("snapshot_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("class_id", sa.Integer(), sa.ForeignKey("ontology_class.id"), nullable=False),
        sa.Column("ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("model_run_id", sa.String(128), nullable=True),
        # The gate's recall shortfall for this class, which is what a label is bought to close.
        sa.Column("deficit", sa.Float(), nullable=True),
        sa.Column("minutes_per_label", sa.Float(), nullable=True),
        sa.Column("inr_per_label", sa.Float(), nullable=True),
        sa.Column("expected_recall_gain", sa.Float(), nullable=True),
        sa.Column("value_per_inr", sa.Float(), nullable=True),
        sa.Column("measured", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("n_timed_reviews", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("minutes_per_label is null or minutes_per_label >= 0",
                           name="ck_label_value_minutes_nonneg"),
        sa.CheckConstraint("inr_per_label is null or inr_per_label >= 0",
                           name="ck_label_value_inr_nonneg"),
        # An unmeasured row must carry its reason, so a null value is never silently a zero.
        sa.CheckConstraint("measured or reason is not null", name="ck_label_value_unmeasured_has_reason"),
    )
    op.create_index("ix_label_value_class_ts", "label_value_snapshot", ["class_id", "ts_ns"])


def downgrade() -> None:
    op.drop_index("ix_label_value_class_ts", table_name="label_value_snapshot")
    op.drop_table("label_value_snapshot")
    op.drop_column("workforce", "rate_inr_per_box")
    op.drop_column("workforce", "rate_inr_per_verdict")
