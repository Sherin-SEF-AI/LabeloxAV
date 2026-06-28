"""R2 data integrity: CHECK constraints on the enum-like columns (object.state/source, app_user.role)
and a non-unique index on frame(session_id, cam_id, ts_ns). A real unique constraint is not added here
because the synthetic corpus already contains colliding (session_id, cam_id, ts_ns) frames; a dedup pass
must run before uniqueness can be enforced. CHECK values are supersets of every value currently in the DB
so the migration applies cleanly.

Revision ID: 0020_constraints
Revises: 0019_dynamics
Create Date: 2026-06-28
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0020_constraints"
down_revision: str | None = "0019_dynamics"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_STATES = ("review", "auto_accept", "accepted", "rejected", "annotate", "submitted")
_SOURCES = ("fused", "auto_accept", "human", "imported", "relabel", "interpolated", "propagated")
_ROLES = ("admin", "reviewer", "annotator")


def _in(col: str, values: tuple[str, ...]) -> str:
    inner = ", ".join(f"'{v}'" for v in values)
    return f"{col} IN ({inner})"


def upgrade() -> None:
    op.create_check_constraint("ck_object_state", "object", _in("state", _STATES))
    op.create_check_constraint("ck_object_source", "object", _in("source", _SOURCES))
    op.create_check_constraint("ck_app_user_role", "app_user", _in("role", _ROLES))
    # Speeds the common per-session frame lookups; intentionally non-unique until the corpus is deduped.
    op.create_index("ix_frame_session_cam_ts", "frame", ["session_id", "cam_id", "ts_ns"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_frame_session_cam_ts", table_name="frame")
    op.drop_constraint("ck_app_user_role", "app_user", type_="check")
    op.drop_constraint("ck_object_source", "object", type_="check")
    op.drop_constraint("ck_object_state", "object", type_="check")
