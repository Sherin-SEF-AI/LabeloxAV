"""Where the champion and a challenger disagree, and which of them a person said was right.

Every model this program has ever built was measured on frozen gold: 164 val images that stopped being
new the day they were sealed. The corpus meanwhile ingests sessions no model is ever compared on, and the
frames that would teach the most, the ones where two models read the same pixels differently, were never
surfaced to anybody. This table is that surface.

A row is one disagreement between two inference runs on one frame, scored so the worst can be ranked
first, and carrying its own directional verdict once a person has ruled. `kind` says what shape the
disagreement has: `champion_miss` and `challenger_miss` are a box one model found and the other did not,
`class_flip` is the same box read as two different classes, `conf_gap` is the same box and class at very
different confidence. `verdict` says who was right, and it is null until somebody says so; there is no
default, because "nobody has looked" and "both were wrong" are not the same fact.

`ScenarioCandidate` was considered and does not fit: it carries one score and one state, no model
references, and no direction. The direction is the entire product here.

Revision ID: 0109_shadow_disagreement
Revises: 0108_origin
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0109_shadow_disagreement"
down_revision = "0108_origin"
branch_labels = None
depends_on = None

KINDS = ("champion_miss", "challenger_miss", "class_flip", "conf_gap")
STATES = ("pending", "queued", "adjudicated", "dismissed")
VERDICTS = ("champion_right", "challenger_right", "both_right", "both_wrong")


def upgrade() -> None:
    op.create_table(
        "shadow_disagreement",
        sa.Column("disagreement_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("sweep_run_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("agent_run.run_id", ondelete="CASCADE"), nullable=False),
        sa.Column("frame_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False),
        sa.Column("champion_run_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("inference_run.run_id", ondelete="CASCADE"), nullable=False),
        sa.Column("challenger_run_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("inference_run.run_id", ondelete="CASCADE"), nullable=False),
        sa.Column("champion_prediction_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("prediction.prediction_id", ondelete="CASCADE"), nullable=True),
        sa.Column("challenger_prediction_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("prediction.prediction_id", ondelete="CASCADE"), nullable=True),
        sa.Column("kind", sa.String(24), nullable=False),
        sa.Column("champion_class_id", sa.Integer(), sa.ForeignKey("ontology_class.id"), nullable=True),
        sa.Column("challenger_class_id", sa.Integer(), sa.ForeignKey("ontology_class.id"), nullable=True),
        sa.Column("iou", sa.Float(), nullable=True),
        sa.Column("conf_gap", sa.Float(), nullable=True),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("bbox", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("state", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("task_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("label_task.task_id", ondelete="SET NULL"), nullable=True),
        sa.Column("verdict", sa.String(24), nullable=True),
        sa.Column("verdict_object_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("object.object_id", ondelete="SET NULL"), nullable=True),
        sa.Column("adjudicated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            "kind in ('" + "','".join(KINDS) + "')", name="ck_shadow_disagreement_kind"),
        sa.CheckConstraint(
            "state in ('" + "','".join(STATES) + "')", name="ck_shadow_disagreement_state"),
        sa.CheckConstraint(
            "verdict is null or verdict in ('" + "','".join(VERDICTS) + "')",
            name="ck_shadow_disagreement_verdict"),
        # A disagreement needs at least one side to point at: a row with neither prediction names nothing.
        sa.CheckConstraint(
            "champion_prediction_id is not null or challenger_prediction_id is not null",
            name="ck_shadow_disagreement_has_a_side"),
    )
    op.create_index("ix_shadow_disagreement_sweep", "shadow_disagreement", ["sweep_run_id", "state"])
    op.create_index("ix_shadow_disagreement_challenger", "shadow_disagreement",
                    ["challenger_run_id", "verdict"])
    op.create_index("ix_shadow_disagreement_frame", "shadow_disagreement", ["frame_id"])


def downgrade() -> None:
    op.drop_index("ix_shadow_disagreement_frame", table_name="shadow_disagreement")
    op.drop_index("ix_shadow_disagreement_challenger", table_name="shadow_disagreement")
    op.drop_index("ix_shadow_disagreement_sweep", table_name="shadow_disagreement")
    op.drop_table("shadow_disagreement")
