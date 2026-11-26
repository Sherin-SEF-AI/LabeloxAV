"""Schema for the scene3d annotation tools, if they need any.

Reserved in sequence so the tools could be built in parallel without forking the revision chain; owned by
agent C (3D tools). A no-op until that work adds tables or columns.

Revision ID: 0128_tools_scene3d
Revises: 0127_tools_video
"""

revision = "0128_tools_scene3d"
down_revision = "0127_tools_video"
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
