"""Session-child FKs cascade on delete + a GIN index on frame.scene.

Four tables (session_index, session_health, frame_group, rig_object) referenced session without ON DELETE
CASCADE, so deleting a session errored on the FK or left orphan rows. Recreate those constraints with CASCADE.
Also add a GIN index on frame.scene (JSONB) so the scene-tag containment queries (weather/time_of_day/road
filters) use an index instead of a full scan.

Revision ID: 0062_cascade_and_scene_gin
Revises: 0061_release_content_fp
Create Date: 2026-07-20
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0062_cascade_and_scene_gin"
down_revision: str | None = "0061_release_content_fp"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FKS = [
    ("session_index", "session_index_session_id_fkey", "session_id"),
    ("session_health", "session_health_session_id_fkey", "session_id"),
    ("frame_group", "frame_group_session_id_fkey", "session_id"),
    ("rig_object", "rig_object_session_id_fkey", "session_id"),
]


def upgrade() -> None:
    for table, fk, col in _FKS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {fk}")
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {fk} FOREIGN KEY ({col}) "
            f"REFERENCES session (session_id) ON DELETE CASCADE"
        )
    op.execute("CREATE INDEX IF NOT EXISTS ix_frame_scene_gin ON frame USING gin (scene)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_frame_scene_gin")
    for table, fk, col in _FKS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {fk}")
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {fk} FOREIGN KEY ({col}) "
            f"REFERENCES session (session_id)"
        )
