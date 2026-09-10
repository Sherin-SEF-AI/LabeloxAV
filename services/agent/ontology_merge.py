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

from sqlalchemy import func, select, update
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


SPLIT_KIND = "ontology_split"
# Objects moved per commit. A split touching a common class is tens of thousands of rows and must not be
# one transaction, for the same reason nothing else here is.
SPLIT_BATCH = 500
# The only states a split may move. `accepted` means a person ruled on that object's class, and a machine
# reassigning it would overwrite a human judgement; those are listed for a person instead.
SPLITTABLE_STATES = ("review", "auto_accept")


class SplitError(Exception):
    """Raised rather than returned, because a caller that ignores this reclassifies the corpus."""


def _matches(obj, rule: dict) -> bool:
    """Whether an object falls on this side of a split, by the rule's attribute predicate.

    Attributes only. A split rule may also carry a VLM prompt for the objects attributes cannot decide,
    and that half deliberately does not run here: asking a model to reclassify tens of thousands of crops
    is a labelling job with its own budget and gate, not something a schema change should do on the way
    past. Objects no rule claims stay where they are and the run counts them.
    """
    attrs = obj.attrs or {}
    for key, want in (rule.get("attrs") or {}).items():
        got = attrs.get(key)
        if isinstance(want, list):
            if got not in want:
                return False
        elif got != want:
            return False
    if (rule.get("min_conf") is not None) and float(obj.conf or 0.0) < float(rule["min_conf"]):
        return False
    return True


async def split_class(db: AsyncSession, *, from_id: int, into: list[dict],
                      to_version: str | None = None, created_by: str | None = None) -> dict:
    """Split one class into several by rule, batch by batch, as one revertible run.

    `into` is a list of `{"to_id": int, "rule": {...}}` in priority order: the first rule an object
    matches wins, so overlapping rules are resolved by the order a person wrote them rather than by
    whichever query returned first.

    Only `review` and `auto_accept` objects move. An `accepted` object is one a person ruled on, and a
    split reassigning it would overwrite a human judgement with a predicate; the run reports how many it
    left alone so the remainder is visible work rather than a silent omission.
    """
    from db.models import ClassMigration

    src = await db.get(OntologyClass, from_id)
    if src is None:
        raise SplitError(f"class {from_id} does not exist")
    if not into:
        raise SplitError("a split needs at least one target class and rule")
    targets = {}
    for spec in into:
        tid = int(spec["to_id"])
        dst = await db.get(OntologyClass, tid)
        if dst is None:
            raise SplitError(f"class {tid} does not exist, so objects would have nowhere to go")
        if tid == from_id:
            raise SplitError("a class cannot be split into itself")
        targets[tid] = dst

    rows = (await db.execute(
        select(Object).where(Object.class_id == from_id,
                             Object.state.in_(SPLITTABLE_STATES)))).scalars().all()
    protected = (await db.execute(
        select(func.count()).select_from(Object)
        .where(Object.class_id == from_id,
               Object.state.notin_(SPLITTABLE_STATES)))).scalar_one()

    run_id = uuid.uuid4()
    changes: dict[str, dict] = {}
    moved: dict[int, int] = dict.fromkeys(targets, 0)
    unclaimed = 0
    for i, obj in enumerate(rows):
        chosen = next((int(s["to_id"]) for s in into if _matches(obj, s.get("rule") or {})), None)
        if chosen is None:
            unclaimed += 1
            continue
        changes[str(obj.object_id)] = {"from_class": from_id}
        obj.class_id = chosen
        moved[chosen] += 1
        if (i + 1) % SPLIT_BATCH == 0:
            await db.commit()

    db.add(AgentRun(
        run_id=run_id, kind=SPLIT_KIND, status="committed",
        scope={"from_id": from_id, "from_name": src.name,
               "into": [{"to_id": t, "to_name": targets[t].name} for t in targets]},
        policy={"into": into, "states": list(SPLITTABLE_STATES)},
        counts={"moved": sum(moved.values()), "per_target": {str(k): v for k, v in moved.items()},
                "unclaimed": unclaimed, "left_human_ruled": int(protected)},
        changes={"objects": changes}, critic={}, created_by=created_by))

    # The rule is recorded because it is the only part of a split that cannot be read off the rows
    # afterwards: which side an object went to leaves no trace of why.
    for spec in into:
        db.add(ClassMigration(
            from_version=src.version, to_version=(to_version or src.version) + "+split",
            from_id=from_id, to_id=int(spec["to_id"]), kind="split",
            rule=spec.get("rule") or {}, run_id=run_id, created_by=created_by))
    await db.commit()

    log.info("ontology.split", run_id=str(run_id), frm=src.name, moved=sum(moved.values()),
             unclaimed=unclaimed, left_human_ruled=int(protected))
    return {"run_id": str(run_id), "from": src.name, "moved": sum(moved.values()),
            "per_target": {targets[k].name: v for k, v in moved.items()},
            "unclaimed": unclaimed, "left_human_ruled": int(protected),
            "note": ("objects a person accepted were not moved: a split reassigning them would overwrite "
                     "a human ruling with a predicate")}
