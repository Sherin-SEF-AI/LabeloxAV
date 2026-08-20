"""60,204 imported objects failed attribute validation forever, on a field that is not an attribute.

The Mapillary importer wrote the source's own class name into `Object.attrs` as `mapillary_label`. The
ontology has never heard of that key, so `validate_attrs` reports "unknown attribute" on every one of those
objects, permanently, and no annotator can fix it because there is nothing wrong with the label.

It only became visible when the reanalyse sweep started reporting attribute validity per frame: 59,520 of
the 66,073 invalid-attribute findings in a 200,000-object sample were this one key, which is 90% of them.
The other 10% are real and were being drowned.

The value itself is worth keeping. It is what the source called the object before the remap, which is the
evidence for auditing a class mapping after the fact. So it moves rather than being deleted, into
`provenance`, where `services/imports/run.py` already writes `import_format`, `import_job` and
`original_name` for exactly this purpose. The adapter now emits it there directly.

Reversible: `downgrade` moves it back, so an install that has read `attrs.mapillary_label` from a stored
export is not stranded. Neither direction touches an object that does not carry the key.
"""

import sqlalchemy as sa
from alembic import op

revision = "0093_mapillary_label_to_prov"
down_revision = "0092_session_project"
branch_labels = None
depends_on = None

# Batched rather than one statement over the whole table. This rewrites two JSONB columns on 60,204 rows,
# and a single UPDATE holds a lock on `object` for its whole duration while the API is serving from it.
_BATCH = 5_000


def _move(conn, src: str, dst: str) -> int:
    """Move `mapillary_label` from one JSONB column to the other. Returns rows touched."""
    moved = 0
    while True:
        res = conn.execute(sa.text(f"""
            with batch as (
                select object_id from object
                where {src} ? 'mapillary_label'
                limit :n
            )
            update object o
               set {dst} = coalesce(o.{dst}, '{{}}'::jsonb)
                           || jsonb_build_object('mapillary_label', o.{src} -> 'mapillary_label'),
                   {src} = o.{src} - 'mapillary_label'
              from batch b
             where o.object_id = b.object_id
        """), {"n": _BATCH})
        if not res.rowcount:
            return moved
        moved += res.rowcount


def upgrade() -> None:
    conn = op.get_bind()
    moved = _move(conn, "attrs", "provenance")
    print(f"0093: moved mapillary_label to provenance on {moved} objects")


def downgrade() -> None:
    conn = op.get_bind()
    moved = _move(conn, "provenance", "attrs")
    print(f"0093: moved mapillary_label back to attrs on {moved} objects")
