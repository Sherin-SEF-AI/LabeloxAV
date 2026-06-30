"""LiDAR BEV frames: frame.lidar holds the point-cloud uri and the BEV projection params so an oriented
box drawn on the rasterized bird's-eye image lifts back to a metric 3D cuboid.

Revision ID: 0024_lidar
Revises: 0023_rot_deg
Create Date: 2026-06-28
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0024_lidar"
down_revision: str | None = "0023_rot_deg"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("frame", sa.Column("lidar", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("frame", "lidar")
