"""Merging a class into another, reversibly, and retiring the ones nothing should propose again.

The custom-class sidecar accumulates whatever an annotator typed. This corpus grew three spellings of one
traffic signal, two of one autorickshaw (one of them misspelled), a `traffic_cone` beside the governed
`cone`, a `small_suv` beside the governed `suv` holding 13,425 objects, and a `test_new_vehicle` left by a
test. None of that is inert: `classify_crop` scores a crop against every class the ontology knows, so the
relabel agent proposed `back_side_of_autorikshasw` for real objects and propagated a typo, and it died
outright on `test_new_vehicle` because that class was proposable and unstorable at once.

Two rules shape this module, and both come from what actually references a class id.

**Only the annotation plane is rewritten.** `object` and `track` are the corpus's current statement about the
world and they move. `prediction` and `eval_patch` are immutable history: a prediction records what a model
said at a moment, and rewriting it would change a measurement that has already been reported. So they keep
pointing at the old class.

**Which means the class row is retired, not deleted.** Those historical rows hold foreign keys to it. What
gets removed is the entry in `ontology/custom_classes.json`, because that is what `get_ontology()` reads and
therefore what the classifier can propose. The row survives so the past stays readable; the name stops being
offered.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Object, OntologyClass, Track

log = get_logger("agent.ontology_merge")

KIND = "ontology_merge"


class MergeError(ValueError):
    """A merge that would lose or corrupt information."""


def _sidecar_path() -> Path:
    from services.autolabel.ontology import _custom_path

    return _custom_path()


def _read_sidecar() -> list[dict]:
    p = _sidecar_path()
    return json.loads(p.read_text()) if p.exists() else []


def _write_sidecar(entries: list[dict]) -> None:
    # indent=2 and sorted keys, matching what `add_custom_class` writes, so the app and this module do not
    # reformat the file against each other on every touch.
    _sidecar_path().write_text(json.dumps(entries, indent=2, sort_keys=True))


async def merge_class(db: AsyncSession, *, from_id: int, to_id: int, created_by: str | None = None) -> dict:
    """Move every object and track from one class to another as one reversible run.

    The target must be a class that still exists, and must not itself be on the way out: merging into
    something being retired in the same pass would leave the objects stranded on a name nothing offers.
    """
    if from_id == to_id:
        raise MergeError("a class cannot be merged into itself")
    src = await db.get(OntologyClass, from_id)
    dst = await db.get(OntologyClass, to_id)
    if src is None:
        raise MergeError(f"class {from_id} does not exist")
    if dst is None:
        raise MergeError(f"class {to_id} does not exist, so the objects would have nowhere to go")

    obj_ids = list((await db.execute(
        select(Object.object_id).where(Object.class_id == from_id))).scalars().all())
    trk_ids = list((await db.execute(
        select(Track.track_id).where(Track.class_id == from_id))).scalars().all())

    run_id = uuid.uuid4()
    # One entry per object, in the shape `revert_run` already understands, so an undo restores the exact
    # prior class rather than guessing from the mapping.
    changes = {str(oid): {"from_class": from_id} for oid in obj_ids}

    if obj_ids:
        await db.execute(update(Object).where(Object.class_id == from_id).values(class_id=to_id))
    if trk_ids:
        await db.execute(update(Track).where(Track.class_id == from_id).values(class_id=to_id))

    db.add(AgentRun(
        run_id=run_id, kind=KIND, status="committed",
        scope={"from_id": from_id, "from_name": src.name, "to_id": to_id, "to_name": dst.name},
        policy={}, counts={"objects": len(obj_ids), "tracks": len(trk_ids)},
        changes={"objects": changes, "tracks": [str(t) for t in trk_ids]},
        critic={}, created_by=created_by))
    await db.commit()
    log.info("ontology.merged", run_id=str(run_id), frm=src.name, to=dst.name,
             objects=len(obj_ids), tracks=len(trk_ids))
    return {"run_id": str(run_id), "from": src.name, "to": dst.name,
            "objects": len(obj_ids), "tracks": len(trk_ids)}


async def revert_merge(db: AsyncSession, run: AgentRun) -> dict:
    """Put every object and track back on the class it came from.

    Unlike an agent relabel this does not skip human-sourced objects. A person labelled something
    `small_suv`; retiring that class is an ontology decision rather than the agent overruling them, so
    undoing the ontology decision has to return their object to the class they chose.
    """
    frm = int(run.scope["from_id"])
    objs = (run.changes or {}).get("objects") or {}
    trks = (run.changes or {}).get("tracks") or []
    n_obj = n_trk = 0
    if objs:
        ids = [uuid.UUID(o) for o in objs]
        res = await db.execute(update(Object).where(Object.object_id.in_(ids)).values(class_id=frm))
        n_obj = res.rowcount or 0
    if trks:
        ids = [uuid.UUID(t) for t in trks]
        res = await db.execute(update(Track).where(Track.track_id.in_(ids)).values(class_id=frm))
        n_trk = res.rowcount or 0
    run.status = "reverted"
    await db.commit()
    log.info("ontology.merge_reverted", run_id=str(run.run_id), objects=n_obj, tracks=n_trk)
    return {"run_id": str(run.run_id), "reverted": n_obj, "tracks": n_trk, "skipped": 0}


def retire_from_sidecar(class_ids: set[int]) -> dict:
    """Stop offering these classes, without deleting the rows history points at.

    Removing the sidecar entry is what actually takes a class out of circulation: `get_ontology()` merges the
    governed YAML with this file, and `classify_crop` scores against whatever that produces. The
    `ontology_class` row stays because `prediction` and `eval_patch` hold foreign keys into it, and those
    records are not ours to rewrite.
    """
    entries = _read_sidecar()
    kept = [e for e in entries if int(e.get("id", -1)) not in class_ids]
    removed = [e["name"] for e in entries if int(e.get("id", -1)) in class_ids]
    _write_sidecar(kept)

    from services.autolabel.ontology import get_ontology

    get_ontology.cache_clear()
    log.info("ontology.retired", removed=sorted(removed), remaining=len(kept))
    return {"removed": sorted(removed), "remaining": len(kept)}


def rename_in_sidecar(class_id: int, new_name: str) -> dict:
    """Correct a class's name in place, keeping its id so nothing that references it breaks."""
    from services.autolabel.ontology import get_ontology, normalize_class_name

    norm = normalize_class_name(new_name)
    if not norm:
        raise MergeError("a class name must contain letters or digits")
    entries = _read_sidecar()
    old = None
    for e in entries:
        if int(e.get("id", -1)) == class_id:
            old = e["name"]
            e["name"] = norm
    if old is None:
        raise MergeError(f"class {class_id} is not in the sidecar")
    _write_sidecar(entries)
    get_ontology.cache_clear()
    log.info("ontology.renamed", class_id=class_id, frm=old, to=norm)
    return {"class_id": class_id, "from": old, "to": norm}
