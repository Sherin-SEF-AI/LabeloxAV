"""camera_calibration: the resolved per-session, per-camera calibration the 3D pipeline reads (intrinsics at
ref_width plus the full 6-DOF camera->ego mount pose and a source provenance), so real calibration overrides
the nominal rig defaults once it is available, and a cuboid's trust follows its calibration source.

Revision ID: 0036_camera_calibration
Revises: 0035_cloud_session
Create Date: 2026-06-30
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0036_camera_calibration"
down_revision: str | None = "0035_cloud_session"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "camera_calibration",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True),
                  sa.ForeignKey("session.session_id", ondelete="CASCADE"), nullable=False),
        sa.Column("cam_id", sa.String(32), nullable=False),
        sa.Column("model", sa.String(16), nullable=False),
        sa.Column("fx", sa.Float(), nullable=False),
        sa.Column("fy", sa.Float(), nullable=False),
        sa.Column("cx", sa.Float(), nullable=False),
        sa.Column("cy", sa.Float(), nullable=False),
        sa.Column("dist", JSONB(), server_default="[]"),
        sa.Column("ref_width", sa.Integer(), nullable=False),
        sa.Column("rpy_deg", JSONB(), server_default="[]"),
        sa.Column("xyz_m", JSONB(), server_default="[]"),
        sa.Column("source", sa.String(16), nullable=False, server_default="nominal"),
        sa.Column("quality", sa.Float(), nullable=False, server_default="0.3"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_camera_calibration_session_cam", "camera_calibration",
                    ["session_id", "cam_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_camera_calibration_session_cam", table_name="camera_calibration")
    op.drop_table("camera_calibration")
