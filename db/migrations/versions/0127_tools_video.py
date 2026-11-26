"""Schema for the video annotation tools: allow `source = 'described'` on objects.

`services/autolabel/describe.py` has always said a described object is written with `source='described'`,
and `POST /frames/{id}/describe-label` says committing goes through "the ordinary object create with
`source='described'`". The CHECK on `object.source` never allowed that value, so nothing could ever be
committed that way. The describe tool in the editor and the session-wide `describe_session` job (agent A's
tool #2) write it now, so the CHECK learns it.

The allowed list is read from the live constraint and extended, rather than restated, because several
reserved tool migrations (0127-0132) run in sequence and another may add its own source value; restating a
literal list here would silently drop theirs or have theirs drop this one.

The downgrade deletes described objects before narrowing the CHECK, as 0108 does for synthetic ones: a
row the narrower constraint forbids cannot stay.

Revision ID: 0127_tools_video
Revises: 0126_tool_jobs
"""

import re

import sqlalchemy as sa
from alembic import op

revision = "0127_tools_video"
down_revision = "0126_tool_jobs"
branch_labels = None
depends_on = None

_NEW = "described"


def _current_sources() -> list[str]:
    bind = op.get_bind()
    defn = bind.execute(sa.text(
        "SELECT pg_get_constraintdef(c.oid) FROM pg_constraint c JOIN pg_class t ON t.oid = c.conrelid "
        "WHERE c.conname = 'ck_object_source' AND t.relname = 'object'")).scalar()
    if not defn:
        return []
    return re.findall(r"'([^']+)'", defn)


def _recreate(values: list[str]) -> None:
    op.drop_constraint("ck_object_source", "object", type_="check")
    quoted = ", ".join(f"'{v}'" for v in values)
    op.create_check_constraint("ck_object_source", "object", f"source IN ({quoted})")


def upgrade() -> None:
    values = _current_sources()
    if values and _NEW not in values:
        _recreate([*values, _NEW])


def downgrade() -> None:
    values = _current_sources()
    if _NEW in values:
        op.execute(sa.text("DELETE FROM object WHERE source = :s").bindparams(s=_NEW))
        _recreate([v for v in values if v != _NEW])
