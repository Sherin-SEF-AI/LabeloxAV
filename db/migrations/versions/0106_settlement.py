"""The settlement engine's representation: a state for machine-settled labels, and the evidence tables.

The machine wrote 99.6% of this corpus's 578,436 objects and was allowed to settle 0.1% of them;
90.5% sits in `state='review'` forever. The operator's decision (2026-09-01): the machine may settle
labels, per class, evidence-gated, sampled-QA'd, revertible, with automatic step-down. This migration
gives that decision a representation. The engine that writes it comes separately; nothing reads or
writes these until it lands.

**Why a state AND a lot table** (both, deliberately). `'settled'` as a sixth object state, because the
three couplings of a record-only design are all live here: the overnight auditor would claw settled
objects back from `auto_accept` nightly, `control_sample` seeding would mix settlement into the gate
precision denominator, and every consumer would need JSONB joins where a state predicate belongs. And
`settlement_lot`, because the acceptance-sampling snapshot - the Wilson decision, the FAR bound, the
sample identity, the run ids, the spot counters - IS the audit trail that makes a settled label
defensible and revocable. A state without the lot is a claim with no evidence; a lot without the
state is evidence nothing enforces.

`'settled'` is never writable through the review API: `services/review_policy.py::state_for` clamps a
human-requested 'settled' to 'accepted', which is the right semantics for free - a person ruling on a
settled object upgrades it to human ground truth. `accepted` keeps meaning "a person ruled",
everywhere; six calibration readers depend on it and a regression test pins that settlement changes
no calibration input.

The CHECK swap is one validating scan, no table rewrite (the 0067 idiom). The downgrade first moves
any settled rows back to 'review' - the conservative direction, the same one the auto-revert uses -
because a downgrade that fails on existing rows is not a downgrade.

`governance_state.settlement_enabled` is born false and stays an explicit opt-in: the kill switch
clears it and release does not restore it.

Revision ID: 0106_settlement
Revises: 0105_control_verdict_at
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0106_settlement"
down_revision = "0105_control_verdict_at"
branch_labels = None
depends_on = None

_WITH = ("state IN ('review', 'auto_accept', 'accepted', 'rejected', 'annotate', 'submitted', "
         "'settled')")
_WITHOUT = "state IN ('review', 'auto_accept', 'accepted', 'rejected', 'annotate', 'submitted')"


def upgrade() -> None:
    op.drop_constraint("ck_object_state", "object", type_="check")
    op.create_check_constraint("ck_object_state", "object", _WITH)

    op.add_column("governance_state",
                  sa.Column("settlement_enabled", sa.Boolean(), nullable=False,
                            server_default="false"))

    op.create_table(
        "settlement_lot",
        sa.Column("lot_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("class_id", sa.Integer(), nullable=False),
        # The champion whose proposals the stratum froze under; pre-attribution objects pool as "legacy".
        sa.Column("model_epoch", sa.String(128), nullable=False),
        sa.Column("population", sa.Integer(), nullable=False),
        sa.Column("tier", sa.String(16), nullable=False),
        # Snapshot of the tier's FAR bound at planning time, so a later config change cannot silently
        # re-justify or invalidate a decision that was made under different rules.
        sa.Column("far_bound", sa.Float(), nullable=False),
        sa.Column("sample_object_ids", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'[]'::jsonb")),
        sa.Column("batch_id", sa.String(64)),
        sa.Column("sample_n", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("defects", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skips", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("topups", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("decision", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        # sampling -> judging -> accepted | rejected | parked; accepted -> settled -> (maybe) reverted
        sa.Column("status", sa.String(16), nullable=False, server_default="sampling"),
        sa.Column("run_ids", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'[]'::jsonb")),
        sa.Column("spot_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("spot_defects", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_by", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("decided_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_settlement_lot_stratum", "settlement_lot",
                    ["class_id", "model_epoch", "status"])

    op.create_table(
        "class_autonomy",
        sa.Column("autonomy_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("class_id", sa.Integer(), nullable=False),
        # 0 propose-only, 1 auto-accept band, 2 settlement auto-apply
        sa.Column("level", sa.Integer(), nullable=False),
        # History rows, one active per class (the ThresholdFit idiom): a step-down is a new row, so the
        # ladder's whole history is readable and a revert is a re-activation, not a lost record.
        sa.Column("active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("basis", postgresql.JSONB(), nullable=False,
                  server_default=sa.text("'{}'::jsonb")),
        sa.Column("set_by", sa.String(128), nullable=False),
        # A human-pinned level is never auto-promoted past.
        sa.Column("pinned", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("cooldown_until", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_class_autonomy_active", "class_autonomy", ["class_id", "active"])

    op.create_table(
        "settlement_spot",
        sa.Column("spot_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("lot_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("settlement_lot.lot_id", ondelete="CASCADE"), nullable=False),
        sa.Column("object_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("object.object_id", ondelete="CASCADE"), nullable=False),
        sa.Column("human_verdict", sa.String(16)),
        sa.Column("verdict_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_settlement_spot_lot", "settlement_spot", ["lot_id", "human_verdict"])


def downgrade() -> None:
    op.drop_index("ix_settlement_spot_lot", table_name="settlement_spot")
    op.drop_table("settlement_spot")
    op.drop_index("ix_class_autonomy_active", table_name="class_autonomy")
    op.drop_table("class_autonomy")
    op.drop_index("ix_settlement_lot_stratum", table_name="settlement_lot")
    op.drop_table("settlement_lot")
    op.drop_column("governance_state", "settlement_enabled")
    # Settled rows go back to review - the conservative direction - so the tightened CHECK can attach.
    op.execute(sa.text("UPDATE object SET state = 'review' WHERE state = 'settled'"))
    op.drop_constraint("ck_object_state", "object", type_="check")
    op.create_check_constraint("ck_object_state", "object", _WITHOUT)
