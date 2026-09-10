"""Where a model came from: its architecture, the model it continued, the model it learned from.

The registry records what a model scored and whether it is champion, and nothing about its ancestry. Two
questions were unanswerable from it: which checkpoint a run continued from, and which teacher a distilled
student was fit against. Both matter as soon as more than one model line exists at once, which is what
shadow mode creates, and the distillation loop cannot express its own result without the second.

`origin` names how the weights came to exist. 'trained' is the default and describes every row that
exists today: a supervised run over labelled frames. 'distilled' is a student fit against a teacher's
soft targets, 'pretrained' a self-supervised stage with no labels, 'imported' a checkpoint that arrived
from outside this system. Backfilling every existing row to 'trained' is accurate rather than convenient:
each one is a detection or segmentation run recorded by the training worker.

Revision ID: 0110_model_lineage
Revises: 0109_shadow_disagreement
"""

import sqlalchemy as sa
from alembic import op

revision = "0110_model_lineage"
down_revision = "0109_shadow_disagreement"
branch_labels = None
depends_on = None

ORIGINS = ("trained", "distilled", "pretrained", "imported")


def upgrade() -> None:
    op.add_column("model_registry", sa.Column("arch", sa.String(64), nullable=True))
    op.add_column("model_registry", sa.Column("parent_version", sa.String(128), nullable=True))
    op.add_column("model_registry", sa.Column("teacher_version", sa.String(128), nullable=True))
    op.add_column("model_registry", sa.Column("origin", sa.String(16), nullable=False,
                                              server_default="trained"))
    op.create_check_constraint(
        "ck_model_registry_origin", "model_registry",
        "origin in ('" + "','".join(ORIGINS) + "')")
    # Self-references rather than hard foreign keys: a parent or teacher can be a checkpoint that was never
    # registered here (a COCO release, a pod-side stage), and a FK would force inventing a registry row for
    # something this system never held. The column is the name it was fit from, present or absent.
    op.create_index("ix_model_registry_parent", "model_registry", ["parent_version"])
    op.create_index("ix_model_registry_teacher", "model_registry", ["teacher_version"])


def downgrade() -> None:
    op.drop_index("ix_model_registry_teacher", table_name="model_registry")
    op.drop_index("ix_model_registry_parent", table_name="model_registry")
    op.drop_constraint("ck_model_registry_origin", "model_registry", type_="check")
    op.drop_column("model_registry", "origin")
    op.drop_column("model_registry", "teacher_version")
    op.drop_column("model_registry", "parent_version")
    op.drop_column("model_registry", "arch")
