"""Tracker identities on predictions, and experiment tracking.

Two gaps that both meant a number could not be computed.

`prediction` carried no identity, so a tracker's output was indistinguishable from a detector's. MOTA, IDF1
and HOTA associate predicted identities with true ones; with no column to put an identity in, there was
nothing to associate and the metrics could never run on real output. Nullable, and that nullness is load
bearing: a detection run leaves it null, and the evaluator refuses rather than scoring a detector as a
tracker that switches identity on every frame.

Experiment tracking existed only as a per-job metric curve, which answers "how did this run go" and not "is
this line of work getting better". Comparing two runs meant reading two mutable job rows and remembering
which hyperparameters went with which. An external tracker is the usual answer and would put the loop's own
history outside the loop, where the promotion gate cannot read it.

Revision ID: 0076_tracks_experiments
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0076_tracks_experiments"
down_revision = "0075_identity_notifs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("prediction", sa.Column("track_id", sa.String(64)))
    # Composite: the tracking evaluator reads one run's identities, never all identities everywhere.
    op.create_index("ix_prediction_track", "prediction", ["run_id", "track_id"])

    op.create_table(
        "experiment",
        sa.Column("experiment_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("task_type", sa.String(32), nullable=False, server_default="detection"),
        sa.Column("description", sa.Text()),
        sa.Column("hypothesis", sa.Text()),
        sa.Column("tags", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("created_by", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_experiment_name", "experiment", ["name"])

    op.create_table(
        "experiment_run",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("experiment_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("experiment.experiment_id", ondelete="CASCADE")),
        # SET NULL, not CASCADE: an experiment's record of what a run scored outlives the operational job
        # row, and losing the comparison when the job is pruned would defeat the point of recording it.
        sa.Column("job_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("training_job.job_id", ondelete="SET NULL")),
        sa.Column("label", sa.String(128)),
        sa.Column("hparams", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("dataset_spec", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("metrics", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("curve", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("gold_id", sa.String(128)),
        sa.Column("baseline_run_id", postgresql.UUID(as_uuid=True)),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        sa.Column("notes", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_experiment_run_exp", "experiment_run", ["experiment_id"])
    op.create_index("ix_experiment_run_job", "experiment_run", ["job_id"])
    # The comparison query: this experiment's runs, in the order they were made.
    op.create_index("ix_experiment_run_exp_started", "experiment_run", ["experiment_id", "started_at"])


def downgrade() -> None:
    op.drop_table("experiment_run")
    op.drop_table("experiment")
    op.drop_index("ix_prediction_track", table_name="prediction")
    op.drop_column("prediction", "track_id")
