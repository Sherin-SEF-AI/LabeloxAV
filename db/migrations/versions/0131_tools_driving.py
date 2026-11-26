"""Schema for the driving annotation tools, if they need any.

Reserved in sequence so the tools could be built in parallel without forking the revision chain; owned by
agent D (driving tools). A no-op until that work adds tables or columns.

Revision ID: 0131_tools_driving
Revises: 0130_tools_quality
"""

revision = "0131_tools_driving"
down_revision = "0130_tools_quality"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
