"""Phase 4 Closed Loop and Governance schema: al_selection, error_candidate, relabel_run, relabel_job,
model_registry, control_sample, drift_metric, assignment, merge_request, audit_decision, governance_state

Revision ID: 0018_closedloop
Revises: 0017_spatial
Create Date: 2026-06-27
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0018_closedloop"
down_revision: str | None = "0017_spatial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PK = dict(primary_key=True, server_default=sa.text("gen_random_uuid()"))


def upgrade() -> None:
    op.create_table(
        "al_selection",
        sa.Column("batch_id", postgresql.UUID(as_uuid=True), **_PK),
        sa.Column("strategy", postgresql.JSONB(), server_default="{}"),
        sa.Column("item_ids", postgresql.ARRAY(sa.Text()), server_default="{}"),
        sa.Column("budget_hours", sa.Float(), server_default="0"),
        sa.Column("expected_value", postgresql.JSONB(), server_default="{}"),
        sa.Column("status", sa.String(16), server_default="open"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_al_selection_status", "al_selection", ["status"])

    op.create_table(
        "error_candidate",
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True), **_PK),
        sa.Column("object_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("object.object_id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(24), nullable=False),
        sa.Column("score", sa.Float(), server_default="0"),
        sa.Column("proposed_label", postgresql.JSONB()),
        sa.Column("detail", postgresql.JSONB(), server_default="{}"),
        sa.Column("status", sa.String(16), server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_error_candidate_status", "error_candidate", ["status"])
    op.create_index("ix_error_candidate_object", "error_candidate", ["object_id"])

    op.create_table(
        "relabel_run",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), **_PK),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("lakefs_branch", sa.String(128)),
        sa.Column("proposed", sa.Integer(), server_default="0"),
        sa.Column("auto_applied", sa.Integer(), server_default="0"),
        sa.Column("routed_to_review", sa.Integer(), server_default="0"),
        sa.Column("regressions_flagged", sa.Integer(), server_default="0"),
        sa.Column("reason", sa.Text()),
        sa.Column("job_id", postgresql.UUID(as_uuid=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "relabel_job",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), **_PK),
        sa.Column("status", sa.String(16), server_default="pending"),
        sa.Column("compute_target", sa.String(16), server_default="local"),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column("session_ids", postgresql.ARRAY(sa.Text()), server_default="{}"),
        sa.Column("ontology_promotion", postgresql.JSONB()),
        sa.Column("stage", sa.String(24)),
        sa.Column("progress", sa.Float(), server_default="0"),
        sa.Column("counts", postgresql.JSONB(), server_default="{}"),
        sa.Column("result", postgresql.JSONB(), server_default="{}"),
        sa.Column("run_id", postgresql.UUID(as_uuid=True)),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_relabel_job_status", "relabel_job", ["status"])

    op.create_table(
        "model_registry",
        sa.Column("model_version", sa.String(128), primary_key=True),
        sa.Column("task", sa.String(32), server_default="detection"),
        sa.Column("gold_metrics", postgresql.JSONB(), server_default="{}"),
        sa.Column("is_champion", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("promoted_from", sa.String(128)),
        sa.Column("dataset_commit", sa.String(128)),
        sa.Column("weights_uri", sa.Text()),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_model_registry_champion", "model_registry", ["task", "is_champion"])

    op.create_table(
        "control_sample",
        sa.Column("sample_id", postgresql.UUID(as_uuid=True), **_PK),
        sa.Column("object_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("object.object_id", ondelete="CASCADE"), nullable=False),
        sa.Column("was_auto_accepted", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("human_verdict", sa.String(16)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_control_sample_verdict", "control_sample", ["human_verdict"])

    op.create_table(
        "drift_metric",
        sa.Column("id", postgresql.UUID(as_uuid=True), **_PK),
        sa.Column("metric", sa.String(24), nullable=False),
        sa.Column("window", postgresql.JSONB(), server_default="{}"),
        sa.Column("value", sa.Float(), server_default="0"),
        sa.Column("breach", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_drift_metric_created", "drift_metric", ["metric", "created_at"])

    op.create_table(
        "assignment",
        sa.Column("assignment_id", postgresql.UUID(as_uuid=True), **_PK),
        sa.Column("item_id", sa.String(128), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("app_user.user_id", ondelete="CASCADE"), nullable=False),
        sa.Column("branch", sa.String(128)),
        sa.Column("status", sa.String(16), server_default="assigned"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_assignment_user", "assignment", ["user_id", "status"])

    op.create_table(
        "merge_request",
        sa.Column("mr_id", postgresql.UUID(as_uuid=True), **_PK),
        sa.Column("title", sa.String(256), nullable=False),
        sa.Column("source_branch", sa.String(128), nullable=False),
        sa.Column("target_branch", sa.String(128), server_default="main"),
        sa.Column("author_id", postgresql.UUID(as_uuid=True)),
        sa.Column("reviewer_id", postgresql.UUID(as_uuid=True)),
        sa.Column("status", sa.String(16), server_default="open"),
        sa.Column("merge_commit", sa.String(128)),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_merge_request_status", "merge_request", ["status"])

    op.create_table(
        "audit_decision",
        sa.Column("audit_id", postgresql.UUID(as_uuid=True), **_PK),
        sa.Column("actor", sa.String(32), server_default="controller"),
        sa.Column("decision", sa.String(48), nullable=False),
        sa.Column("subject", sa.String(128)),
        sa.Column("rationale", postgresql.JSONB(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_audit_created", "audit_decision", ["created_at"])
    op.create_index("ix_audit_actor", "audit_decision", ["actor"])

    op.create_table(
        "governance_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("loop_enabled", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("auto_accept_enabled", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("auto_promote_enabled", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("champion_version", sa.String(128)),
        sa.Column("paused_reason", sa.Text()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    # seed the singleton governance row
    op.execute("INSERT INTO governance_state (id, loop_enabled, auto_accept_enabled, auto_promote_enabled) "
               "VALUES (1, true, true, true) ON CONFLICT (id) DO NOTHING")


def downgrade() -> None:
    for t in ("governance_state", "audit_decision", "merge_request", "assignment", "drift_metric",
              "control_sample", "model_registry", "relabel_job", "relabel_run", "error_candidate", "al_selection"):
        op.drop_table(t)
