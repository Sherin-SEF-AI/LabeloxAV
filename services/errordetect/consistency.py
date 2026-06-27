"""Consistency-based label-error detection (M4.1). Track-level: an object whose class differs from its
track's majority is a likely flip (M2.0 tracks). Cross-camera: an object sharing a rig identity (M3.1
rig_track_id) but labeled a different class in another synchronized view is inconsistent. Both are strong,
cheap error signals over existing structure."""

from __future__ import annotations

from collections import Counter

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object
from services.autolabel.ontology import get_ontology

log = get_logger("ed_consistency")


def _majority(class_ids: list[int]) -> tuple[int, int]:
    cnt = Counter(class_ids)
    cid, n = cnt.most_common(1)[0]
    return cid, n


async def detect_consistency(db: AsyncSession, session_id: str | None = None) -> list[dict]:
    onto = get_ontology()
    out: list[dict] = []

    # track-level class flips
    tq = select(Object.object_id, Object.track_id, Object.class_id).where(Object.track_id.isnot(None))
    if session_id:
        tq = tq.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == session_id)
    tracks: dict = {}
    for oid, tid, cid in (await db.execute(tq)).all():
        tracks.setdefault(tid, []).append((str(oid), cid))
    for tid, members in tracks.items():
        if len(members) < 3:
            continue
        maj, n_maj = _majority([c for _, c in members])
        if n_maj == len(members):
            continue
        for oid, c in members:
            if c != maj:
                out.append({"object_id": oid, "kind": "track_inconsistent",
                            "score": round(n_maj / len(members), 4),
                            "proposed_label": {"class_id": maj, "class_name": onto.by_id(maj).name},
                            "detail": {"track_id": str(tid), "given_class": onto.by_id(c).name,
                                       "track_majority": onto.by_id(maj).name, "track_len": len(members)}})

    # cross-camera inconsistency over rig identities
    rq = select(Object.object_id, Object.rig_track_id, Object.class_id).where(Object.rig_track_id.isnot(None))
    if session_id:
        rq = rq.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == session_id)
    rigs: dict = {}
    for oid, rid, cid in (await db.execute(rq)).all():
        rigs.setdefault(rid, []).append((str(oid), cid))
    for rid, members in rigs.items():
        if len(members) < 2 or len({c for _, c in members}) < 2:
            continue
        maj, n_maj = _majority([c for _, c in members])
        for oid, c in members:
            if c != maj:
                out.append({"object_id": oid, "kind": "cross_cam_inconsistent",
                            "score": round(n_maj / len(members), 4),
                            "proposed_label": {"class_id": maj, "class_name": onto.by_id(maj).name},
                            "detail": {"rig_track_id": str(rid), "given_class": onto.by_id(c).name,
                                       "rig_majority": onto.by_id(maj).name, "n_views": len(members)}})

    log.info("ed.consistency", flagged=len(out))
    return out
