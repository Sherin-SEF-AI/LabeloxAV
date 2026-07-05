"""Data Engine M0: complete the canonical spine for the VERDYX (evaluation), FORGYX (benchmark, deployment)
and ORACLYX (pseudo-label consensus) planes, plus additive SANYX/CALYX columns on the existing health and
calibration tables. All additive; no existing column is altered or dropped.

Revision ID: 0049_data_engine_spine
Revises: 0048_scenegraph_vlm
Create Date: 2026-07-05
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0049_data_engine_spine"
down_revision: str | None = "0048_scenegraph_vlm"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # VERDYX: first-class per-slice evaluation + champion/challenger verdict
    op.create_table(
        "evaluation",
        sa.Column("eval_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("model_version", sa.String(128),
                  sa.ForeignKey("model_registry.model_version", ondelete="CASCADE"), nullable=False),
        sa.Column("release_commit", sa.String(128),
                  sa.ForeignKey("dataset_commit.commit_id", ondelete="SET NULL"), nullable=True),
        sa.Column("gold_id", sa.String(128), sa.ForeignKey("gold_set.gold_id", ondelete="SET NULL"), nullable=True),
        sa.Column("per_slice", JSONB(), nullable=False, server_default="{}"),
        sa.Column("failure_clusters", JSONB(), nullable=False, server_default="{}"),
        sa.Column("aggregate", JSONB(), nullable=False, server_default="{}"),
        sa.Column("verdict", sa.String(16), nullable=False, server_default="needs_review"),
        sa.Column("challenger_of", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_evaluation_model", "evaluation", ["model_version"])

    # FORGYX: per (model, target) latency/accuracy/power
    op.create_table(
        "benchmark",
        sa.Column("benchmark_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("model_version", sa.String(128),
                  sa.ForeignKey("model_registry.model_version", ondelete="CASCADE"), nullable=False),
        sa.Column("target", sa.String(32), nullable=False),
        sa.Column("latency_ms", JSONB(), nullable=False, server_default="{}"),
        sa.Column("throughput_fps", sa.Float(), nullable=True),
        sa.Column("power_w", sa.Float(), nullable=True),
        sa.Column("accuracy_ref", UUID(as_uuid=True),
                  sa.ForeignKey("evaluation.eval_id", ondelete="SET NULL"), nullable=True),
        sa.Column("per_layer_uri", sa.Text(), nullable=True),
        sa.Column("pareto_rank", sa.Integer(), nullable=True),
        sa.Column("artifact_uri", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_benchmark_model_target", "benchmark", ["model_version", "target"])

    # FORGYX: deployable, verified artifact per target with lineage to release + verdict + benchmark
    op.create_table(
        "deployment",
        sa.Column("deployment_id", UUID(as_uuid=True), primary_key=True),
        sa.Column("model_version", sa.String(128),
                  sa.ForeignKey("model_registry.model_version", ondelete="CASCADE"), nullable=False),
        sa.Column("target", sa.String(32), nullable=False),
        sa.Column("artifact_uri", sa.Text(), nullable=False),
        sa.Column("export_format", sa.String(16), nullable=False),
        sa.Column("release_commit", sa.String(128),
                  sa.ForeignKey("dataset_commit.commit_id", ondelete="SET NULL"), nullable=True),
        sa.Column("verdict_ref", UUID(as_uuid=True),
                  sa.ForeignKey("evaluation.eval_id", ondelete="SET NULL"), nullable=True),
        sa.Column("benchmark_ref", UUID(as_uuid=True),
                  sa.ForeignKey("benchmark.benchmark_id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.String(12), nullable=False, server_default="built"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_deployment_model", "deployment", ["model_version"])

    # ORACLYX: offline-fusion consensus over a fused object (side table, hot object row untouched)
    op.create_table(
        "pseudo_label",
        sa.Column("object_id", UUID(as_uuid=True),
                  sa.ForeignKey("object.object_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("consensus", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("consensus_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("voters", JSONB(), nullable=False, server_default="{}"),
        sa.Column("fusion_run_id", sa.String(128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # SANYX: overall health score + tri-state decision on the existing per-session health row
    op.add_column("session_health", sa.Column("score", sa.Float(), nullable=True))
    op.add_column("session_health", sa.Column("decision", sa.String(12), nullable=True))

    # CALYX: SE(3) drift delta + severity on the existing calibration validation row
    op.add_column("calibration_validation", sa.Column("drift_delta", JSONB(), nullable=True))
    op.add_column("calibration_validation", sa.Column("severity", sa.String(16), nullable=True))


def downgrade() -> None:
    op.drop_column("calibration_validation", "severity")
    op.drop_column("calibration_validation", "drift_delta")
    op.drop_column("session_health", "decision")
    op.drop_column("session_health", "score")
    op.drop_table("pseudo_label")
    op.drop_index("ix_deployment_model", table_name="deployment")
    op.drop_table("deployment")
    op.drop_index("ix_benchmark_model_target", table_name="benchmark")
    op.drop_table("benchmark")
    op.drop_index("ix_evaluation_model", table_name="evaluation")
    op.drop_table("evaluation")
