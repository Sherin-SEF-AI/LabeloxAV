"""Static-camera sessions: add session.pack_id and make session.vehicle_id nullable.

SEC-M2 of the multi-domain refactor. A session is no longer necessarily a moving-vehicle drive: a
static-camera (CCTV) session has no ego vehicle. pack_id routes a session to its domain pack; it defaults to
'av' so every existing row backfills to the AV pack, and vehicle_id becomes nullable so a static-camera
session can omit it. Both changes are additive and reversible; the AV ingestion path still sets vehicle_id.

Revision ID: 0068_session_pack_id
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0068_session_pack_id"
down_revision = "0067_vlm_review_source"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("session", sa.Column("pack_id", sa.String(length=32), server_default="av", nullable=True))
    # Backfill any existing rows explicitly (server_default covers new rows; this covers rows written by a
    # concurrent path before the default applied).
    op.execute("UPDATE session SET pack_id = 'av' WHERE pack_id IS NULL")
    op.alter_column("session", "vehicle_id", existing_type=sa.String(length=64), nullable=True)


def downgrade() -> None:
    # Restore NOT NULL on vehicle_id, defaulting any static-camera rows (which have none) to a sentinel so the
    # constraint can be re-applied without data loss.
    op.execute("UPDATE session SET vehicle_id = 'unknown' WHERE vehicle_id IS NULL")
    op.alter_column("session", "vehicle_id", existing_type=sa.String(length=64), nullable=False)
    op.drop_column("session", "pack_id")
