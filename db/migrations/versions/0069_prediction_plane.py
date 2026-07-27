"""The immutable prediction plane: inference_run + prediction.

Measurement was scoring live corpus rows as predictions, but human review mutates those rows in place, so
every confirmed-correct detection was erased from the prediction population. Predictions now live in their own
append-only plane that review never touches. See db/models.py InferenceRun / Prediction and docs/MEASUREMENT.md.

Additive only.

Revision ID: 0069_prediction_plane
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0069_prediction_plane"
down_revision = "0068_session_pack_id"
branch_labels = None
depends_on = None

_UUID = postgresql.UUID(as_uuid=True)
_JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "inference_run",
        sa.Column("run_id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("model_version", sa.String(length=128),
                  sa.ForeignKey("model_registry.model_version", ondelete="CASCADE"), nullable=False),
        sa.Column("gold_id", sa.String(length=128), nullable=True),
        sa.Column("frame_count", sa.Integer(), server_default="0"),
        sa.Column("params", _JSONB, server_default="{}"),
        sa.Column("code_sha", sa.String(length=40), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="running"),
    )
    op.create_index("ix_inference_run_model_gold", "inference_run", ["model_version", "gold_id"])

    op.create_table(
        "prediction",
        sa.Column("prediction_id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", _UUID, sa.ForeignKey("inference_run.run_id", ondelete="CASCADE"), nullable=False),
        sa.Column("frame_id", _UUID, sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=False),
        sa.Column("class_id", sa.Integer(), sa.ForeignKey("ontology_class.id"), nullable=False),
        sa.Column("bbox", postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column("conf", sa.Float(), nullable=False),
        sa.Column("conf_calibrated", sa.Float(), nullable=True),
        sa.Column("rot_deg", sa.Float(), server_default="0.0"),
        sa.Column("mask_uri", sa.Text(), nullable=True),
        sa.Column("mask_encoding", sa.String(length=16), nullable=True),
        sa.Column("cuboid_3d", _JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_prediction_run", "prediction", ["run_id"])
    op.create_index("ix_prediction_frame", "prediction", ["frame_id"])
    op.create_index("ix_prediction_run_frame", "prediction", ["run_id", "frame_id"])


def downgrade() -> None:
    op.drop_index("ix_prediction_run_frame", table_name="prediction")
    op.drop_index("ix_prediction_frame", table_name="prediction")
    op.drop_index("ix_prediction_run", table_name="prediction")
    op.drop_table("prediction")
    op.drop_index("ix_inference_run_model_gold", table_name="inference_run")
    op.drop_table("inference_run")
