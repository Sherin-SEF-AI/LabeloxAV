"""Cross-view track handoff and consistency (M-MC.4): chain the per-instant rig identities (RigObject, M-MC.2)
into a rig track that follows one physical object across time and across cameras, then check that track for a
consistent class. When a car is tracked in the front camera and handed off to the right camera as it passes,
those are one rig track; if the front view calls it a car and the right view calls it a truck, that is a
cross_cam_inconsistent error the review queue should see.

The temporal handoff reuses the existing per-camera tracks (Object.track_id from M2.0): two rig identities in
different frame groups belong to the same rig track when they share a per-camera track, i.e. the same object
track passes through both. No calibration is needed for the handoff (it is identity, not geometry); the
consistency check is pure label agreement.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from uuid import UUID

from sqlalchemy import delete, select, update

from core.logging import get_logger
from db.models import ErrorCandidate, Frame, Object, RigObject
from db.session import get_sessionmaker

log = get_logger("multicam.rigtrack")


async def _rig_members(db, session_id: UUID) -> tuple[list[RigObject], dict]:
    """Every rig object in the session and, per member object, its (track_id, class_id, cam, ts_ns)."""
    rigs = (await db.execute(
        select(RigObject).where(RigObject.session_id == session_id).order_by(RigObject.created_at))).scalars().all()
    member_ids = {m for r in rigs for m in (r.member_object_ids or [])}
    info: dict = {}
    if member_ids:
        rows = (await db.execute(
            select(Object.object_id, Object.track_id, Object.class_id, Frame.cam_id, Frame.ts_ns)
            .join(Frame, Frame.frame_id == Object.frame_id)
            .where(Object.object_id.in_(list(member_ids))))).all()
        info = {oid: {"track_id": tid, "class_id": cid, "cam": cam, "ts_ns": int(ts)}
                for oid, tid, cid, cam, ts in rows}
    return list(rigs), info


async def build_rig_tracks(session_id: UUID) -> dict:
    """Assign a rig_track_id to each rig object by union-find: two rig objects join when they share a per-camera
    track (the same tracked object seen at two instants). Rig objects with no tracked members stand alone."""
    maker = get_sessionmaker()
    async with maker() as db:
        rigs, info = await _rig_members(db, session_id)
        parent = {r.rig_object_id: r.rig_object_id for r in rigs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # index rig objects by the per-camera tracks their members belong to
        by_track: dict = defaultdict(list)
        for r in rigs:
            for m in (r.member_object_ids or []):
                tid = info.get(m, {}).get("track_id")
                if tid is not None:
                    by_track[tid].append(r.rig_object_id)
        for rig_ids in by_track.values():
            for other in rig_ids[1:]:
                a, b = find(rig_ids[0]), find(other)
                if a != b:
                    parent[a] = b

        comp_track: dict = {}
        for r in rigs:
            root = find(r.rig_object_id)
            comp_track.setdefault(root, uuid.uuid4())
            r.rig_track_id = comp_track[root]
        await db.commit()
        n_tracks = len(set(comp_track.values()))
    log.info("multicam.rig_tracks_built", session_id=str(session_id), rig_objects=len(rigs), tracks=n_tracks)
    return {"session_id": str(session_id), "rig_objects": len(rigs), "rig_tracks": n_tracks}


async def rig_tracks(session_id: UUID) -> dict:
    """Rig tracks for the session: one row per track with its instant count, cameras, time span, voted class,
    and whether it is inconsistent (members disagree on class across cameras or time)."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()

    def name(cid):
        try:
            return onto.by_id(int(cid)).name
        except Exception:  # noqa: BLE001
            return str(cid)

    maker = get_sessionmaker()
    async with maker() as db:
        rigs, info = await _rig_members(db, session_id)
        by_track: dict = defaultdict(list)
        for r in rigs:
            if r.rig_track_id is not None:
                by_track[r.rig_track_id].append(r)
        out = []
        for tid, members in by_track.items():
            classes, cams, ts = [], set(), []
            for r in members:
                for m in (r.member_object_ids or []):
                    d = info.get(m)
                    if d:
                        classes.append(d["class_id"])
                        cams.add(d["cam"])
                        ts.append(d["ts_ns"])
            voted = Counter([c for c in classes if c is not None]).most_common(1)
            out.append({"rig_track_id": str(tid), "instants": len(members), "cameras": sorted(cams),
                        "ts_start": min(ts) if ts else None, "ts_end": max(ts) if ts else None,
                        "class_name": name(voted[0][0]) if voted else None,
                        "inconsistent": len({c for c in classes if c is not None}) > 1})
        out.sort(key=lambda t: (not t["inconsistent"], -t["instants"]))
        return {"session_id": str(session_id), "n_tracks": len(out), "tracks": out}


async def rig_track_timeline(session_id: UUID, rig_track_id: UUID) -> dict:
    """The ordered instants of one rig track: each frame group's rig object with its cameras, members, and the
    per-member class, so the UI can draw the track's handoff across cameras over time."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()

    def name(cid):
        try:
            return onto.by_id(int(cid)).name
        except Exception:  # noqa: BLE001
            return str(cid)

    maker = get_sessionmaker()
    async with maker() as db:
        rigs = (await db.execute(
            select(RigObject).where(RigObject.session_id == session_id,
                                    RigObject.rig_track_id == rig_track_id))).scalars().all()
        _, info = await _rig_members(db, session_id)
        instants = []
        for r in rigs:
            members = [{"object_id": str(m), "cam": info.get(m, {}).get("cam"),
                        "class_name": name(info.get(m, {}).get("class_id"))} for m in (r.member_object_ids or [])]
            ts = [info.get(m, {}).get("ts_ns") for m in (r.member_object_ids or []) if info.get(m)]
            instants.append({"rig_object_id": str(r.rig_object_id), "group_id": str(r.group_id),
                             "ts_ns": min(ts) if ts else None, "class_name": name(r.class_id) if r.class_id is not None else None,
                             "conflict": bool(r.conflict), "cameras": sorted({mm["cam"] for mm in members if mm["cam"]}),
                             "members": members})
        instants.sort(key=lambda i: (i["ts_ns"] is None, i["ts_ns"] or 0))
        return {"rig_track_id": str(rig_track_id), "n_instants": len(instants), "instants": instants}


async def check_consistency(session_id: UUID) -> dict:
    """Flag cross-view label disagreement on rig tracks. For each track the members' classes are voted; any
    member whose class differs from the vote becomes a cross_cam_inconsistent error candidate (proposing the
    voted class), so the existing review queue can confirm or reject it. Rebuilds tracks first."""
    await build_rig_tracks(session_id)
    maker = get_sessionmaker()
    async with maker() as db:
        rigs, info = await _rig_members(db, session_id)
        by_track: dict = defaultdict(list)
        for r in rigs:
            if r.rig_track_id is not None:
                by_track[r.rig_track_id].append(r)

        candidates = []
        for tid, members in by_track.items():
            per_obj = {}
            for r in members:
                for m in (r.member_object_ids or []):
                    d = info.get(m)
                    if d and d["class_id"] is not None:
                        per_obj[m] = d
            classes = [d["class_id"] for d in per_obj.values()]
            if len(set(classes)) < 2:
                continue  # consistent track
            voted, n_voted = Counter(classes).most_common(1)[0]
            for oid, d in per_obj.items():
                if d["class_id"] != voted:
                    candidates.append({"object_id": str(oid), "kind": "cross_cam_inconsistent",
                                       "score": round(1.0 - n_voted / len(classes), 3),
                                       "proposed_label": {"class_id": int(voted)},
                                       "detail": {"rig_track_id": str(tid), "voted_class": int(voted),
                                                  "this_class": int(d["class_id"]), "cam": d["cam"],
                                                  "cameras": sorted({x["cam"] for x in per_obj.values()})}})

        # keep the strongest per object, replace this session's prior pending cross_cam candidates
        best: dict = {}
        for c in candidates:
            k = c["object_id"]
            if k not in best or c["score"] > best[k]["score"]:
                best[k] = c
        sess_obj_ids = select(Object.object_id).join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == session_id)
        await db.execute(delete(ErrorCandidate).where(
            ErrorCandidate.kind == "cross_cam_inconsistent", ErrorCandidate.status == "pending",
            ErrorCandidate.object_id.in_(sess_obj_ids)))
        for c in best.values():
            db.add(ErrorCandidate(object_id=UUID(c["object_id"]), kind=c["kind"], score=c["score"],
                                  proposed_label=c["proposed_label"], detail=c["detail"], status="pending"))
        await db.commit()
    log.info("multicam.consistency", session_id=str(session_id), inconsistent=len(best))
    return {"session_id": str(session_id), "n_tracks": len(by_track), "inconsistent_objects": len(best),
            "candidates": list(best.values())}
