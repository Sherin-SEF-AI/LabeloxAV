"""EvalPatch points at the prediction plane: add run_id and prediction_id.

Evaluation now scores an InferenceRun's Prediction rows, not live corpus state. A true/false positive patch is
a prediction (prediction_id); a false negative is the missed gold object (object_id). model_version is derived
from the run, no longer caller-supplied (the misattribution source). object_id is already nullable.

Additive only.

Revision ID: 0070_eval_patch_run
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0070_eval_patch_run"
down_revision = "0069_prediction_plane"
branch_labels = None
depends_on = None

_UUID = postgresql.UUID(as_uuid=True)


def upgrade() -> None:
    op.add_column("eval_patch", sa.Column("run_id", _UUID, nullable=True))
    op.add_column("eval_patch", sa.Column("prediction_id", _UUID, nullable=True))
    op.create_foreign_key("fk_eval_patch_run", "eval_patch", "inference_run",
                          ["run_id"], ["run_id"], ondelete="CASCADE")
    op.create_foreign_key("fk_eval_patch_prediction", "eval_patch", "prediction",
                          ["prediction_id"], ["prediction_id"], ondelete="CASCADE")
    op.create_index("ix_eval_patch_run", "eval_patch", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_eval_patch_run", table_name="eval_patch")
    op.drop_constraint("fk_eval_patch_prediction", "eval_patch", type_="foreignkey")
    op.drop_constraint("fk_eval_patch_run", "eval_patch", type_="foreignkey")
    op.drop_column("eval_patch", "prediction_id")
    op.drop_column("eval_patch", "run_id")
