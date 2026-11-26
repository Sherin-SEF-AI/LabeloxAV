"""Schema for the quality annotation tools: allow `source = 'consensus'` on objects.

The consensus merge (services/labelops/agreement.py `merge_replica_group`, tool #11) writes one object per
physical thing that enough independent annotators and model votes agreed on. That object is neither any
annotator's (`human`) nor the autolabeler's (`fused`), and it must stay distinguishable from both: a reader
that counts human verdicts must not count a vote tally as one, and an undo of the merge must find exactly the
rows the merge wrote. So it gets its own source value, and the CHECK on `object.source` learns it.

The allowed list is read from the live constraint and extended rather than restated, as 0127 does for
`described`: several reserved tool migrations (0127-0132) run in sequence, and restating a literal list here
would silently drop a value another one added.

Nothing else is needed. The VLM label checks (#13) live in the existing `machine_verdict` table (judge
`vlm_reason`), the error explanations (#12) are built from the candidate on read, and the rare-scene ranking
(#14) is the result of a `tool_job`.

The downgrade deletes consensus objects before narrowing the CHECK, as 0127 does for described ones: a row
the narrower constraint forbids cannot stay.

Revision ID: 0130_tools_quality
Revises: 0129_tools_agents
"""

import re

import sqlalchemy as sa
from alembic import op

revision = "0130_tools_quality"
down_revision = "0129_tools_agents"
branch_labels = None
depends_on = None

_NEW = "consensus"


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
