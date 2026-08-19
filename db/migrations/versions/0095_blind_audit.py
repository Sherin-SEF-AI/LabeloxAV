"""Capture-recapture: the tables that let recall be measured against something other than itself.

Gold recall is recall against a denominator somebody already found, and the two ways of finding an object
cost very different amounts of a person's attention: confirming a machine box is one click, drawing a
missed one is thirty seconds. The gold set is therefore biased toward what the model already sees, and
every recall number the engine has ever reported is an overestimate by an unknown amount. The immutable
prediction plane fixed the numerator. This fixes the denominator.

Three tables:

  blind_audit          a set of frames a human labels with predictions and existing objects withheld
  blind_audit_frame    one frame in that audit, and the three capture counts it contributed
  recapture_estimate   the population estimate, per stratum and per class, keyed on (run_id, gold_id)

Plus `object.blind_audit_id`, which marks a box as an independent observation rather than an ordinary
review label. The distinction is load-bearing: an ordinary label is usually a confirmed machine box and
carries no information at all about what the model missed.

`recapture_estimate` stores unmeasurable slices as rows with `measured = false` and a reason rather than
omitting them, because a missing row reads as "not computed" and this needs to say "computed, and the
answer is that we cannot tell". Its unique constraint is NULLS NOT DISTINCT: null means pooled in both key
columns, and under the SQL default the pooled row could be inserted twice while every other slice was
correctly protected.

Nothing here backfills. There are no blind audits, and inventing rows would be exactly the fabricated
denominator this set out to remove.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0095_blind_audit"
down_revision = "0094_unscoped_attrs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "blind_audit",
        sa.Column("audit_id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("gold_id", sa.String(128)),
        sa.Column("job_id", sa.UUID()),
        sa.Column("n_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("stratify_by", sa.String(32), nullable=False, server_default="density"),
        sa.Column("strata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("score_thr", sa.Float(), nullable=False, server_default="0.25"),
        sa.Column("iou_thr", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("status", sa.String(16), nullable=False, server_default="seeded"),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("scored_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["run_id"], ["inference_run.run_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["job_id"], ["label_job.job_id"], ondelete="SET NULL"),
    )
    op.create_index("ix_blind_audit_run", "blind_audit", ["run_id"])
    op.create_index("ix_blind_audit_gold", "blind_audit", ["gold_id"])
    op.create_index("ix_blind_audit_job", "blind_audit", ["job_id"])

    op.create_table(
        "blind_audit_frame",
        sa.Column("audit_frame_id", sa.UUID(), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("audit_id", sa.UUID(), nullable=False),
        sa.Column("frame_id", sa.UUID(), nullable=False),
        sa.Column("stratum", sa.String(64), nullable=False, server_default="all"),
        sa.Column("labeled_at", sa.DateTime(timezone=True)),
        # Nullable, and null is not zero: an unlabelled frame is not a frame where the human found nothing.
        sa.Column("n_both", sa.Integer()),
        sa.Column("n_model_only", sa.Integer()),
        sa.Column("n_human_only", sa.Integer()),
        sa.ForeignKeyConstraint(["audit_id"], ["blind_audit.audit_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["frame_id"], ["frame.frame_id"], ondelete="CASCADE"),
        sa.UniqueConstraint("audit_id", "frame_id", name="uq_blind_audit_frame"),
    )
    op.create_index("ix_blind_audit_frame_audit", "blind_audit_frame", ["audit_id"])
    # The blindness guard asks "is this frame under an audit" on every editor frame fetch. The unique
    # constraint's index leads with audit_id and cannot answer that, so this is what keeps the guard off
    # the sequential-scan path on the editor's hot path.
    op.create_index("ix_blind_audit_frame_frame", "blind_audit_frame", ["frame_id"])

    op.create_table(
        "recapture_estimate",
        sa.Column("estimate_id", sa.UUID(), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("audit_id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("gold_id", sa.String(128)),
        sa.Column("stratum", sa.String(64)),
        sa.Column("class_id", sa.Integer()),
        sa.Column("n_both", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("n_model_only", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("n_human_only", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("measured", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("reason", sa.Text()),
        sa.Column("population", sa.Float()),
        sa.Column("population_lo", sa.Float()),
        sa.Column("population_hi", sa.Float()),
        sa.Column("variance", sa.Float()),
        sa.Column("model_recall", sa.Float()),
        sa.Column("recall_lo", sa.Float()),
        sa.Column("recall_hi", sa.Float()),
        sa.Column("human_recall", sa.Float()),
        sa.Column("gold_recall", sa.Float()),
        sa.Column("estimator", sa.String(32), nullable=False, server_default="chapman-lp-v1"),
        sa.Column("n_strata_pooled", sa.Integer()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["audit_id"], ["blind_audit.audit_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["inference_run.run_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_recapture_estimate_run_gold", "recapture_estimate", ["run_id", "gold_id"])
    op.create_index("ix_recapture_estimate_audit", "recapture_estimate", ["audit_id"])
    # Not expressible through sa.UniqueConstraint's portable arguments; Postgres 15+.
    op.execute("ALTER TABLE recapture_estimate ADD CONSTRAINT uq_recapture_estimate_slice "
               "UNIQUE NULLS NOT DISTINCT (audit_id, stratum, class_id)")

    op.add_column("object", sa.Column("blind_audit_id", sa.UUID(), nullable=True))
    op.create_foreign_key("fk_object_blind_audit", "object", "blind_audit",
                          ["blind_audit_id"], ["audit_id"], ondelete="SET NULL")
    # Partial: the column is null on all 576,469 existing objects and on nearly every future one, so an
    # index over the nulls would be most of the corpus and would answer no query anybody asks.
    op.create_index("ix_object_blind_audit", "object", ["blind_audit_id"],
                    postgresql_where=sa.text("blind_audit_id IS NOT NULL"))


def downgrade() -> None:
    op.drop_index("ix_object_blind_audit", table_name="object")
    op.drop_constraint("fk_object_blind_audit", "object", type_="foreignkey")
    op.drop_column("object", "blind_audit_id")

    op.drop_index("ix_recapture_estimate_audit", table_name="recapture_estimate")
    op.drop_index("ix_recapture_estimate_run_gold", table_name="recapture_estimate")
    op.drop_table("recapture_estimate")

    op.drop_index("ix_blind_audit_frame_frame", table_name="blind_audit_frame")
    op.drop_index("ix_blind_audit_frame_audit", table_name="blind_audit_frame")
    op.drop_table("blind_audit_frame")

    op.drop_index("ix_blind_audit_job", table_name="blind_audit")
    op.drop_index("ix_blind_audit_gold", table_name="blind_audit")
    op.drop_index("ix_blind_audit_run", table_name="blind_audit")
    op.drop_table("blind_audit")
