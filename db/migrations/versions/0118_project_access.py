"""Who may see a project's sessions: a visibility per project, and its members.

Every session already belongs to a project (session.project_id, migration 0092), but nothing decided who could
see one: every signed-in person saw everything. That suits one small team and nothing else: a lab that takes
on a customer's drives, or a sensitive collection, had no way to keep them from the rest of the team.

`label_project.visibility` is "team" (everyone signed in, which is what every project is today and stays) or
"members" (its members and administrators only). `project_member` says who is in a project. Membership only
matters for "members" projects, so an install that never uses the setting behaves exactly as before.

Revision ID: 0118_project_access
Revises: 0117_autolabel_job_queue
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0118_project_access"
down_revision = "0117_autolabel_job_queue"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("label_project", sa.Column("visibility", sa.String(16), nullable=False, server_default="team"))
    op.create_check_constraint("ck_label_project_visibility", "label_project",
                               "visibility IN ('team', 'members')")
    op.create_table(
        "project_member",
        sa.Column("project_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("label_project.project_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("app_user.user_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("added_by", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("app_user.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_project_member_user", "project_member", ["user_id"])
    # Sessions that arrived after 0092 with no project (ingest never set one) join the default project, so the
    # first members-only project does not leave them in a state no rule describes.
    op.execute("""
        UPDATE session SET project_id = (SELECT project_id FROM label_project WHERE name = 'default')
        WHERE project_id IS NULL AND EXISTS (SELECT 1 FROM label_project WHERE name = 'default')
    """)


def downgrade() -> None:
    op.drop_index("ix_project_member_user", table_name="project_member")
    op.drop_table("project_member")
    op.drop_constraint("ck_label_project_visibility", "label_project", type_="check")
    op.drop_column("label_project", "visibility")
