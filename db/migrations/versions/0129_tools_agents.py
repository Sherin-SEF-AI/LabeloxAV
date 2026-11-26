"""Schema for the agents annotation tools, if they need any.

Reserved in sequence so the tools could be built in parallel without forking the revision chain; owned by
agent F (language edits and autopilot). A no-op until that work adds tables or columns.

Revision ID: 0129_tools_agents
Revises: 0128_tools_scene3d
"""

revision = "0129_tools_agents"
down_revision = "0128_tools_scene3d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
