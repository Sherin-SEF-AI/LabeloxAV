"""Auto-accept thresholds that were measured rather than picked.

`gate.auto_accept = 0.95` and `gate.safety_auto_accept = 0.99` are constants. The gate calls them
calibrated precision floors, which would only be true if somebody had measured the precision at those
scores. Nobody did. Two classes at the same nominal 0.95 can sit at very different real precisions, and a
recalibration moves both without moving either constant.

This table holds a per-class operating point fitted from measured outcomes by
core/accel/np_threshold.py: the smallest score whose false-accept rate among accepted detections stays
under the class's bound, with a bootstrap interval on the threshold itself.

One row per (fit_id, class_id). A fit is replaced wholesale, never patched per class, because a threshold
set assembled from two evaluations is not an operating point.

A class that could not be fitted gets a row with `measured = false` and a reason. The gate has to
distinguish "this class earned no threshold" from "nobody looked", and fall back loudly rather than
silently.

`active` defaults false. Fitted is not in force: services/autolabel/gold_calibrate.py already sets the
precedent of fitting, reporting whether the result is trustworthy, and leaving activation to a person.
Nothing here changes a live threshold on its own.
"""

import sqlalchemy as sa
from alembic import op

revision = "0096_threshold_fit"
down_revision = "0095_blind_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "threshold_fit",
        sa.Column("row_id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("fit_id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("gold_id", sa.String(128)),
        sa.Column("class_id", sa.Integer(), nullable=False),
        sa.Column("class_name", sa.String(128), nullable=False),
        # Which confidence column the pairs were read from. A threshold fitted on calibrated confidence and
        # applied to raw confidence is not conservative, it is arbitrary.
        sa.Column("score_field", sa.String(24), nullable=False, server_default="conf"),
        sa.Column("alpha", sa.Float(), nullable=False),
        sa.Column("measured", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("reason", sa.Text()),
        sa.Column("threshold", sa.Float()),
        sa.Column("threshold_lo", sa.Float()),
        sa.Column("threshold_hi", sa.Float()),
        sa.Column("far_at", sa.Float()),
        sa.Column("accept_rate", sa.Float()),
        sa.Column("n_accept", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("n_pairs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("n_positive", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("n_boot_fit", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("config_threshold", sa.Float()),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["run_id"], ["inference_run.run_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("fit_id", "class_id", name="uq_threshold_fit_class"),
    )
    # The gate's lookup: the active fit for a model version. Partial, because all but one fit per model is
    # inactive and indexing the inactive rows would answer no query the gate asks.
    op.create_index("ix_threshold_fit_model", "threshold_fit", ["model_version", "active"],
                    postgresql_where=sa.text("active"))
    op.create_index("ix_threshold_fit_run", "threshold_fit", ["run_id"])
    op.create_index("ix_threshold_fit_fit", "threshold_fit", ["fit_id"])


def downgrade() -> None:
    op.drop_index("ix_threshold_fit_fit", table_name="threshold_fit")
    op.drop_index("ix_threshold_fit_run", table_name="threshold_fit")
    op.drop_index("ix_threshold_fit_model", table_name="threshold_fit")
    op.drop_table("threshold_fit")
