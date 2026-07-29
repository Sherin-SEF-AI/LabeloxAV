"""Compare two sealed dataset commits, down to the objects and the classes.

A comparison already existed in `services/export/snapshots.py` and stopped at the headline: object count,
ontology version, slice spec. That answers "is it bigger" and not "what changed", which is the question a
buyer and a release manager both actually ask. Which objects entered, which left, and which classes moved
were unavailable without unpacking two archives and diffing them by hand, which nobody does, so in practice
versions shipped with no statement of what moved.

The comparison is over object ids and class distribution rather than over files. Two exports of the same
objects in different formats are the same dataset, and a byte diff of their archives would report every file
as changed while nothing about the data had.

An ontology change is called out separately and loudly. When the class vocabulary moves under a dataset, a
class count that appears to have grown may only have been renamed, so a diff that reported the counts
without the vocabulary change would actively mislead.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import DatasetCommit

log = get_logger("dataset_diff")


async def _class_histogram(db: AsyncSession, object_ids: list[str]) -> dict[str, int]:
    """Per-class counts for a set of objects, by name rather than id.

    By name because ids are only meaningful within one ontology version, and the whole point of this
    comparison is that the two sides may not share one.
    """
    import uuid as _uuid

    from db.models import Object
    from services.autolabel.ontology import get_ontology

    if not object_ids:
        return {}
    onto = get_ontology()
    rows = (await db.execute(
        select(Object.class_id).where(
            Object.object_id.in_([_uuid.UUID(o) for o in object_ids])))).scalars().all()
    out: dict[str, int] = {}
    for cid in rows:
        try:
            name = onto.by_id(int(cid)).name
        except Exception:  # noqa: BLE001 - a class the current ontology has dropped still gets counted
            name = f"class_{int(cid)}"
        out[name] = out.get(name, 0) + 1
    return out


async def deep_diff_commits(db: AsyncSession, commit_a: str, commit_b: str, *,
                            sample: int = 20) -> dict:
    """What changed between two sealed commits.

    `sample` bounds the example lists. Returning every added id on a hundred-thousand-object diff would make
    the response the size of the dataset; the counts are exact and the examples are illustrative, which is
    stated in the output so nobody mistakes one for the other.
    """
    a = await db.get(DatasetCommit, commit_a)
    b = await db.get(DatasetCommit, commit_b)
    missing = [c for c, v in ((commit_a, a), (commit_b, b)) if v is None]
    if missing:
        raise ValueError(f"dataset commit(s) not found: {missing}")

    ids_a = set(await _object_ids(db, a))
    ids_b = set(await _object_ids(db, b))
    added, removed, kept = ids_b - ids_a, ids_a - ids_b, ids_a & ids_b

    hist_a = await _class_histogram(db, sorted(ids_a))
    hist_b = await _class_histogram(db, sorted(ids_b))
    class_delta = {}
    for name in sorted(set(hist_a) | set(hist_b)):
        before, after = hist_a.get(name, 0), hist_b.get(name, 0)
        if before != after:
            class_delta[name] = {"before": before, "after": after, "delta": after - before}

    spec_a, spec_b = dict(a.slice_spec or {}), dict(b.slice_spec or {})
    spec_delta = {k: {"a": spec_a.get(k), "b": spec_b.get(k)}
                  for k in sorted(set(spec_a) | set(spec_b))
                  if spec_a.get(k) != spec_b.get(k)}

    ontology_changed = a.ontology_version != b.ontology_version
    result = {
        "a": _commit_dict(a), "b": _commit_dict(b),
        "objects": {
            "a": len(ids_a), "b": len(ids_b),
            "added": len(added), "removed": len(removed), "unchanged": len(kept),
            # Exact counts, illustrative examples. Said out loud so the two are not confused.
            "example_added": sorted(added)[:sample],
            "example_removed": sorted(removed)[:sample],
            "examples_are_a_sample": len(added) > sample or len(removed) > sample,
        },
        "classes": {"delta": class_delta,
                    "gained": sorted(set(hist_b) - set(hist_a)),
                    "lost": sorted(set(hist_a) - set(hist_b))},
        "slice_spec_delta": spec_delta,
        "ontology": {
            "a": a.ontology_version, "b": b.ontology_version, "changed": ontology_changed,
            # The warning matters: under a vocabulary change a class that looks like it grew may only have
            # been renamed, and a count-only diff would actively mislead.
            "warning": ("the ontology version changed between these commits, so a class that appears to "
                        "have gained or lost objects may only have been renamed"
                        if ontology_changed else None),
        },
        "formats": {"a": spec_a.get("formats", []), "b": spec_b.get("formats", []),
                    "added": sorted(set(spec_b.get("formats", [])) - set(spec_a.get("formats", []))),
                    "removed": sorted(set(spec_a.get("formats", [])) - set(spec_b.get("formats", [])))},
    }
    log.info("dataset.diffed", a=commit_a[:12], b=commit_b[:12],
             added=len(added), removed=len(removed))
    return result


async def _object_ids(db: AsyncSession, commit: DatasetCommit) -> list[str]:
    """The object ids a commit covers.

    Read from the commit's own record where it kept them; otherwise re-derived from its slice spec, which is
    what makes an older commit comparable rather than unavailable.
    """
    stored = getattr(commit, "object_ids", None)
    if stored:
        return [str(o) for o in stored]

    from services.export.dataset import SliceSpec, fetch_records

    try:
        spec = SliceSpec(**(commit.slice_spec or {}))
    except Exception:  # noqa: BLE001
        return []
    records = await fetch_records(spec)
    return [str(r.object_id) for r in records]


def _commit_dict(c: DatasetCommit) -> dict:
    return {"commit_id": c.commit_id, "name": (c.slice_spec or {}).get("name"),
            "object_count": c.object_count, "ontology_version": c.ontology_version,
            "created_at": c.created_at.isoformat() if c.created_at else None}


async def lineage_chain(db: AsyncSession, name: str, limit: int = 20) -> dict:
    """A dataset's versions in order, each with the delta from the one before it.

    The list a release manager reads: not "here are twenty commits" but "here is how this dataset moved".
    """
    rows = (await db.execute(
        select(DatasetCommit).order_by(DatasetCommit.created_at.desc()).limit(200))).scalars().all()
    chain = [c for c in rows if (c.slice_spec or {}).get("name") == name][:limit]
    chain.reverse()

    out = []
    for i, c in enumerate(chain):
        entry = {**_commit_dict(c), "delta": None}
        if i > 0:
            prev = chain[i - 1]
            entry["delta"] = {"objects": int(c.object_count or 0) - int(prev.object_count or 0),
                              "from": prev.commit_id}
        out.append(entry)
    return {"name": name, "versions": out}
