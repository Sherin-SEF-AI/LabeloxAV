"""Schema for the sensors annotation tools, if they need any.

Reserved in sequence so the tools could be built in parallel without forking the revision chain; owned by
agent E (sensors, reconstruction, synthetic). A no-op until that work adds tables or columns.

Revision ID: 0132_tools_sensors
Revises: 0131_tools_driving
"""

revision = "0132_tools_sensors"
down_revision = "0131_tools_driving"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
