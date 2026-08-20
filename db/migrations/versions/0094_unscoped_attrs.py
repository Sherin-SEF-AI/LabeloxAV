"""7,477 objects that are not traffic signals carry a `signal_state`, and it exports.

62,366 attribute values on 16,223 objects are attributes their own class cannot have. One autorickshaw
carried signal_state, signal_kind, signal_mount, signal_arrow, marking_state, articulated and helmet at
once. None of those are observations of anything. The VLM verification path handed the model every attribute
in the ontology on every crop and then validated the reply without a class id, which is the argument that
turns `validate_attrs`' applicability check on. The values were checked against their enums; whether the
attribute belonged on the object was never checked at all.

`services/autolabel/paths/path_c_qwen3vl.py` now scopes the schema it asks for and filters the reply against
the class the object ends up with, so no new ones are written. This is the 16,223 already stored.

**Moved, not deleted.** They go to `provenance.unscoped_attrs`, so:

  - `attrs` stops presenting a fabrication as an annotation, which is what matters, because attrs is what
    exports and what an annotator reads as fact;
  - the values survive for anyone auditing what the VLM path used to emit, which is the evidence that this
    happened and the only record of its scale;
  - `downgrade` puts them back exactly.

Deliberately not a delete. A bulk write over 16,223 objects' labels should be reversible, and "the model
said this while looking at the crop" is a real fact about the model even when it is not a fact about the
object.
"""

import sqlalchemy as sa
from alembic import op

revision = "0094_unscoped_attrs"
down_revision = "0093_mapillary_label_to_prov"
branch_labels = None
depends_on = None

_BATCH = 2_000


def _scope_by_l1() -> dict:
    """The ontology's per-subclass attribute allowlist, read from the sidecar rather than hardcoded here.

    A migration that pasted the allowlist would freeze it at today's shape, and the next scope change would
    make this file quietly disagree with the ontology it is enforcing.
    """
    from services.autolabel.ontology import get_ontology

    return dict(get_ontology().attribute_scope)


def _move(conn, batch: int = _BATCH) -> int:
    """Move every out-of-scope attribute into provenance.unscoped_attrs. Returns objects touched."""
    scope = _scope_by_l1()
    if not scope:
        return 0
    touched = 0
    for l1, allowed in scope.items():
        while True:
            res = conn.execute(sa.text("""
                with batch as (
                    select o.object_id,
                           (select jsonb_object_agg(k, v) from jsonb_each(o.attrs) e(k, v)
                             where not (k = any(:allowed))) as offending,
                           (select coalesce(jsonb_object_agg(k, v), '{}'::jsonb)
                              from jsonb_each(o.attrs) e(k, v) where k = any(:allowed)) as keep
                      from object o
                      join ontology_class c on c.id = o.class_id
                     where c.l1 = :l1
                       and exists (select 1 from jsonb_object_keys(o.attrs) k
                                    where not (k = any(:allowed)))
                     limit :n
                )
                update object o
                   set attrs = b.keep,
                       provenance = coalesce(o.provenance, '{}'::jsonb)
                                    || jsonb_build_object('unscoped_attrs',
                                         coalesce(o.provenance -> 'unscoped_attrs', '{}'::jsonb)
                                         || b.offending)
                  from batch b
                 where o.object_id = b.object_id
            """), {"l1": l1, "allowed": list(allowed), "n": batch})
            if not res.rowcount:
                break
            touched += res.rowcount
    return touched


def _restore(conn, batch: int = _BATCH) -> int:
    """Put them back into attrs and drop the provenance key."""
    touched = 0
    while True:
        res = conn.execute(sa.text("""
            with batch as (
                select object_id from object
                 where provenance ? 'unscoped_attrs'
                 limit :n
            )
            update object o
               set attrs = coalesce(o.attrs, '{}'::jsonb) || (o.provenance -> 'unscoped_attrs'),
                   provenance = o.provenance - 'unscoped_attrs'
              from batch b
             where o.object_id = b.object_id
        """), {"n": batch})
        if not res.rowcount:
            return touched
        touched += res.rowcount


def upgrade() -> None:
    moved = _move(op.get_bind())
    print(f"0094: moved out-of-scope attributes into provenance on {moved} objects")


def downgrade() -> None:
    back = _restore(op.get_bind())
    print(f"0094: restored out-of-scope attributes onto {back} objects")
