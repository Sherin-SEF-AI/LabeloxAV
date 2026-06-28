"""Oriented 2D boxes: object.rot_deg, the rotation in degrees about the box centre. Additive: bbox stays
the unrotated AABB (export/IPM/dynamics unchanged), this angle rides on top for consumers that want an
oriented box.

Revision ID: 0023_rot_deg
Revises: 0022_cuboid3d
Create Date: 2026-06-28
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0023_rot_deg"
down_revision: Union[str, None] = "0022_cuboid3d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("object", sa.Column("rot_deg", sa.Float(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("object", "rot_deg")
