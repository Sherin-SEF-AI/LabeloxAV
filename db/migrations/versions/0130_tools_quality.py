"""Schema for the quality annotation tools, if they need any.

Reserved in sequence so the tools could be built in parallel without forking the revision chain; owned by
agent B (quality tools). A no-op until that work adds tables or columns.

Revision ID: 0130_tools_quality
Revises: 0129_tools_agents
"""

revision = "0130_tools_quality"
down_revision = "0129_tools_agents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
