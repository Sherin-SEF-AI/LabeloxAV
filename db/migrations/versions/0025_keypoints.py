"""Keypoints/skeleton: object.keypoints, a COCO-style {"skeleton","points":[[x,y,v],...]} in image px for
pedestrian/cyclist pose annotation. Additive, nullable.

Revision ID: 0025_keypoints
Revises: 0024_lidar
Create Date: 2026-06-28
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0025_keypoints"
down_revision: Union[str, None] = "0024_lidar"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("object", sa.Column("keypoints", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("object", "keypoints")
