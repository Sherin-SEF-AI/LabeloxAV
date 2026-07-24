"""Labeling operations: projects, tasks, assignable jobs, and issue threads.

The work-management layer LabeloxAV lacked. ImportJob and TrainingJob are compute jobs; none of them model a
person picking up a bounded batch of frames, moving it through annotation, validation and acceptance, and
being measured on it. label_job is that unit.

Jobs reference frames by id rather than copying them, so a job is a view over the corpus and cannot drift
from it. stage and state are separate columns on purpose: stage is where in the pipeline the work sits,
state is how far along it is within that stage, and collapsing them loses "in validation, not yet started".

Additive only.

Revision ID: 0064_labelops
Revises: 0063_explore_foundation
Create Date: 2026-07-24
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0064_labelops"
down_revision: str | None = "0063_explore_foundation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UUID = postgresql.UUID(as_uuid=True)
_JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    op.create_table(
        "label_project",
        sa.Column("project_id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(length=120), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("modality", sa.String(length=16), nullable=False, server_default="image"),
        sa.Column("label_config", _JSONB, server_default="{}"),
        sa.Column("honeypot_frac", sa.Float(), nullable=False, server_default="0"),
        sa.Column("min_honeypot_accuracy", sa.Float(), nullable=False, server_default="0.9"),
        sa.Column("gold_id", sa.String(length=128), nullable=True),
        sa.Column("created_by", _UUID, sa.ForeignKey("app_user.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "label_task",
        sa.Column("task_id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("project_id", _UUID, sa.ForeignKey("label_project.project_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("session_id", _UUID, sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=True),
        sa.Column("slice_id", _UUID, sa.ForeignKey("curation_slice.slice_id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("predicate", _JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_label_task_project", "label_task", ["project_id"])

    op.create_table(
        "label_job",
        sa.Column("job_id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("task_id", _UUID, sa.ForeignKey("label_task.task_id", ondelete="CASCADE"), nullable=False),
        sa.Column("frame_ids", _JSONB, server_default="[]"),
        sa.Column("assignee_id", _UUID, sa.ForeignKey("app_user.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("stage", sa.String(length=12), nullable=False, server_default="annotation"),
        sa.Column("state", sa.String(length=12), nullable=False, server_default="new"),
        sa.Column("honeypot_frame_ids", _JSONB, server_default="[]"),
        sa.Column("honeypot_accuracy", sa.Float(), nullable=True),
        sa.Column("honeypot_detail", _JSONB, server_default="{}"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_label_job_task", "label_job", ["task_id"])
    op.create_index("ix_label_job_assignee", "label_job", ["assignee_id", "state"])

    op.create_table(
        "issue",
        sa.Column("issue_id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_id", _UUID, sa.ForeignKey("label_job.job_id", ondelete="CASCADE"), nullable=True),
        sa.Column("frame_id", _UUID, sa.ForeignKey("frame.frame_id", ondelete="CASCADE"), nullable=True),
        sa.Column("object_id", _UUID, sa.ForeignKey("object.object_id", ondelete="CASCADE"), nullable=True),
        sa.Column("region", _JSONB, nullable=True),
        sa.Column("kind", sa.String(length=24), nullable=False, server_default="comment"),
        sa.Column("status", sa.String(length=12), nullable=False, server_default="open"),
        sa.Column("created_by", _UUID, sa.ForeignKey("app_user.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("resolved_by", _UUID, sa.ForeignKey("app_user.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_issue_frame", "issue", ["frame_id", "status"])
    op.create_index("ix_issue_job", "issue", ["job_id", "status"])

    op.create_table(
        "issue_comment",
        sa.Column("comment_id", _UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("issue_id", _UUID, sa.ForeignKey("issue.issue_id", ondelete="CASCADE"), nullable=False),
        sa.Column("author_id", _UUID, sa.ForeignKey("app_user.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_issue_comment_issue", "issue_comment", ["issue_id"])


def downgrade() -> None:
    op.drop_table("issue_comment")
    op.drop_table("issue")
    op.drop_table("label_job")
    op.drop_table("label_task")
    op.drop_table("label_project")
