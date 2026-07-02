"""Temporal auto-repair: a tracked object should not change class frame to frame, so when a track is
overwhelmingly one class with a few odd-one-out frames, those outliers are almost certainly the tracker or
detector slipping -- relabel them to the track majority automatically.

Only self-heals when the majority is strong (>= min_majority of the track) and the track is long enough to
trust; anything ambiguous is left for a human (detect_consistency still surfaces it in the fix queue). Only
machine frames are relabeled, never a human's. One reversible AgentRun records each object's original class
so revert restores it exactly.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Frame, Object
from services.errordetect.consistency import _majority

log = get_logger("agent.temporal_repair")

# Only these classes legitimately form a track. If a track's majority is static infrastructure (tree,
# bus_shelter, pole), the track itself is corrupt -- it linked different objects -- so relabeling to the
# majority would inject an error. Those are left for a human; only movable-majority flips self-heal.
_MOVABLE = {"two_wheeler", "three_wheeler", "four_wheeler", "heavy", "vru", "animal"}


async def _tracks(db: AsyncSession, session_id: str | None):
    q = select(Object.object_id, Object.track_id, Object.class_id, Object.source).where(Object.track_id.isnot(None))
    if session_id:
        q = q.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == uuid.UUID(session_id))
    tracks: dict = {}
    for oid, tid, cid, src in (await db.execute(q)).all():
        tracks.setdefault(tid, []).append((oid, int(cid), src))
    return tracks


async def plan_temporal_repair(db: AsyncSession, session_id: str | None = None, *, min_majority: float = 0.8,
                               min_len: int = 3) -> dict:
    """Dry-run: which class-flip outliers would be relabeled to their track majority. No writes."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    tracks = await _tracks(db, session_id)
    items = []
    counts = {"tracks": len(tracks), "flipped_tracks": 0, "relabels": 0}
    for tid, members in tracks.items():
        if len(members) < min_len:
            continue
        classes = [c for _, c, _ in members]
        maj, n_maj = _majority(classes)
        if n_maj == len(members):
            continue
        counts["flipped_tracks"] += 1
        frac = n_maj / len(members)
        if frac < min_majority:
            continue
        try:
            if onto.by_id(maj).l1 not in _MOVABLE:  # corrupt track (static-majority): leave for a human
                counts["skipped_static"] = counts.get("skipped_static", 0) + 1
                continue
        except Exception:  # noqa: BLE001
            continue
        for oid, cid, src in members:
            if cid != maj and src != "human":
                counts["relabels"] += 1
                items.append({"object_id": str(oid), "track_id": str(tid), "from_class": cid, "to_class": maj,
                              "from_name": onto.by_id(cid).name, "to_name": onto.by_id(maj).name,
                              "majority": round(frac, 3), "track_len": len(members)})
    return {"session_id": session_id or "corpus", "counts": counts, "items": items}


async def commit_temporal_repair(db: AsyncSession, session_id: str | None = None, *, min_majority: float = 0.8,
                                 min_len: int = 3, created_by: str | None = None) -> dict:
    """Relabel the strong-majority outliers to the track class as one reversible run (revert restores class)."""
    plan = await plan_temporal_repair(db, session_id, min_majority=min_majority, min_len=min_len)
    run_id = uuid.uuid4()
    changes: dict[str, dict] = {}
    for item in plan["items"]:
        obj = await db.get(Object, uuid.UUID(item["object_id"]))
        if obj is None or obj.source == "human" or int(obj.class_id) == item["to_class"]:
            continue
        changes[item["object_id"]] = {"from_class": item["from_class"]}
        obj.class_id = item["to_class"]
        obj.version = (obj.version or 0) + 1
        prov = dict(obj.provenance or {})
        prov["agent_run_id"] = str(run_id)
        prov.setdefault("agent_temporal", {})["fixed"] = f"{item['from_name']} -> {item['to_name']} (track {item['majority']})"
        obj.provenance = prov
    db.add(AgentRun(run_id=run_id, kind="temporal_repair", scope={"session_id": session_id},
                    status="committed", policy={"min_majority": min_majority, "min_len": min_len},
                    counts=plan["counts"], changes=changes, critic={}, created_by=created_by))
    await db.commit()
    log.info("agent.temporal_repair.commit", session_id=session_id, run_id=str(run_id), relabeled=len(changes))
    return {"run_id": str(run_id), "session_id": session_id or "corpus", "relabeled": len(changes), "counts": plan["counts"]}
