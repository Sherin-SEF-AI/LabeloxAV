"""Undo for a bulk review, and why the generic revert cannot do it.

Bulk review writes one `Review` row per object, which is the audit trail and answers "who changed this and
what was it before". What it did not write was anything tying the fifty objects together, so a fifty-object
mistake was fifty manual reversals, each one requiring the operator to remember what the value had been. The
correction dialog inherits this: it exists to apply one decision widely, which is exactly the operation most
worth being able to take back in one move.

`AgentRun` is already the repo's revertible unit and `services/agent/runs.py revert_run` already restores
`from_class`, `from_state`, `from_source` and `from_attrs`. It cannot be reused as is, for a reason its own
comment states: it refuses to touch anything whose `source` is `human`, which is right when undoing an
agent's work over a person's, and wrong here because a human review is what set `source = "human"` in the
first place. Every object in the batch would be skipped.

So this kind gets its own revert, the way `ontology_merge` and `cleanup_sweep` already do. The ownership
check that keeps it safe is not the source column but the run id stamped in provenance: an object is
restored only while it still carries THIS run's id, so anything edited afterwards, by a person or by a later
batch, is left alone and reported as skipped rather than silently rolled back.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Object

log = get_logger("review_batch")

KIND = "review_batch"


def change_record(obj: Object) -> dict:
    """What has to be remembered about one object for the batch to be undoable.

    Captured before the edit. `from_source` matters as much as `from_class`: a machine label promoted to
    `human` by a batch that turns out to be wrong must go back to being a machine label, or the corpus quietly
    gains human-authored rows nobody authored.
    """
    return {"from_class": int(obj.class_id), "from_state": obj.state,
            "from_source": obj.source, "from_attrs": dict(obj.attrs or {})}


async def record_batch(db: AsyncSession, changes: dict[str, dict], *, policy: dict | None = None,
                       created_by: str | None = None, commit: bool = True) -> str | None:
    """Tie an applied batch together so it can be taken back. Returns the run id, or None for an empty batch.

    Stamps each object with the run id, which is what the revert checks ownership against.

    `commit=False` lets the caller land the stamp in the same transaction as the edit it describes. That is
    the correct way to use this: the review router used to commit the edits and only then call this, on the
    reasoning that a failed edit must not leave a run claiming objects it never changed - true, but it
    bought that by opening a window where the process could die between the two and leave the batch
    permanently un-revertible, because revert_batch keys ownership on precisely this stamp. One transaction
    satisfies both: a failed edit rolls the stamp back with it, and a committed edit is always revertible.
    """
    if not changes:
        return None
    run_id = uuid.uuid4()
    for oid in changes:
        obj = await db.get(Object, uuid.UUID(oid))
        if obj is None:
            continue
        prov = dict(obj.provenance or {})
        prov["agent_run_id"] = str(run_id)
        prov["review_batch"] = True
        obj.provenance = prov
    db.add(AgentRun(run_id=run_id, kind=KIND, scope={}, status="committed",
                    policy=policy or {}, counts={"objects": len(changes)},
                    changes=changes, critic={}, created_by=created_by or "review"))
    if commit:
        await db.commit()
    log.info("review_batch.recorded", run_id=str(run_id), objects=len(changes))
    return str(run_id)


async def revert_batch(db: AsyncSession, run: AgentRun) -> dict:
    """Put a bulk review back, including the objects it made human-sourced.

    Ownership is the stamped run id rather than the source column, so a later edit by anybody wins and is
    counted as skipped. That is the same protection the generic revert gets from its `source == "human"`
    check, expressed in the one way that still works when the run itself is the reason the row says human.
    """
    reverted = skipped = 0
    for oid, ch in (run.changes or {}).items():
        obj = await db.get(Object, uuid.UUID(oid))
        if obj is None:
            skipped += 1
            continue
        prov = dict(obj.provenance or {})
        if str(prov.get("agent_run_id")) != str(run.run_id):
            skipped += 1
            continue
        if "from_class" in ch:
            obj.class_id = ch["from_class"]
        if "from_state" in ch:
            obj.state = ch["from_state"]
        if "from_source" in ch:
            obj.source = ch["from_source"]
        if "from_attrs" in ch:
            obj.attrs = ch["from_attrs"]
        obj.version = (obj.version or 0) + 1
        prov.pop("agent_run_id", None)
        prov.pop("review_batch", None)
        obj.provenance = prov
        reverted += 1
    run.status = "reverted"
    run.reverted_at = datetime.now(UTC)
    await db.commit()
    log.info("review_batch.reverted", run_id=str(run.run_id), reverted=reverted, skipped=skipped)
    return {"run_id": str(run.run_id), "reverted": reverted, "skipped": skipped}
