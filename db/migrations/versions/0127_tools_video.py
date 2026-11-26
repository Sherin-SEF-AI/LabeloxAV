"""Schema for the video annotation tools, if they need any.

Reserved in sequence so the tools could be built in parallel without forking the revision chain; owned by
agent A (video and editor tools). A no-op until that work adds tables or columns.

Revision ID: 0127_tools_video
Revises: 0126_tool_jobs
"""

revision = "0127_tools_video"
down_revision = "0126_tool_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
