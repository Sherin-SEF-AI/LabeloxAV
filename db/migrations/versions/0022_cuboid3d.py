"""R4.3 3D: object.cuboid_3d, an optional ego-frame 3D box {center,size,yaw} that lifts the dataset to
3D where a cuboid label exists and drives a real (non-placeholder) nuScenes/KITTI 3D export.

Revision ID: 0022_cuboid3d
Revises: 0021_object_version
Create Date: 2026-06-28
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0022_cuboid3d"
down_revision: Union[str, None] = "0021_object_version"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("object", sa.Column("cuboid_3d", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("object", "cuboid_3d")
