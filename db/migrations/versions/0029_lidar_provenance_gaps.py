"""LiDAR Phase 3 provenance hardening (from the adversarial review): record the calibration that placed a
traversability product and the input calibrations of an aggregated map, so the provenance walk is complete.

Revision ID: 0029_lidar_provenance_gaps
Revises: 0028_lidar_scene_intelligence
Create Date: 2026-06-29
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0029_lidar_provenance_gaps"
down_revision: str | None = "0028_lidar_scene_intelligence"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("traversability", sa.Column("calibration_version", sa.String(64), nullable=True))
    op.add_column("aggregated_map", sa.Column("input_calibrations", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("aggregated_map", "input_calibrations")
    op.drop_column("traversability", "calibration_version")
