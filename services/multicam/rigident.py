"""Rig identity and linked selection (M-MC.2, Tier 1): bind the per-camera Object rows that are the same
physical thing seen in different views at one instant into a single RigObject. This tier needs NO calibration
(it never projects geometry): a reviewer links objects manually, and a DINOv3 appearance assist only proposes
candidate links, it never applies them (agents propose, gates dispose). The rig object carries a voted class
across its members; when members disagree it is flagged as a conflict and its non-human members are routed to
review, because a cross-view class disagreement is strong evidence one of the labels is wrong.

Contrast with services/multicam/associate.py, which is the calibration-gated appearance auto-association that
writes rig_track_id across time. Here linking is per-instant (within one frame group) and explicit.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from uuid import UUID

import numpy as np
from sqlalchemy import select, update

from core.logging import get_logger
from db.models import FrameGroup, Object, ObjectEmbedding, RigObject
from db.session import get_sessionmaker

log = get_logger("multicam.rigident")


def _vote(class_ids: list[int]) -> tuple[int | None, bool]:
    """Voted class across members and whether they conflict (more than one distinct class)."""
    ids = [c for c in class_ids if c is not None]
    if not ids:
        return None, False
    voted = Counter(ids).most_common(1)[0][0]
    return int(voted), len(set(ids)) > 1


async def _group_objects(db, group: FrameGroup) -> list[dict]:
    """All non-rejected objects in a group's frames, with their camera, class, rig link, and DINOv3 vector."""
    frame_of_cam = group.frame_ids or {}
    cam_of_frame = {UUID(fid): cam for cam, fid in frame_of_cam.items()}
    if not cam_of_frame:
        return []
    rows = (await db.execute(
        select(Object.object_id, Object.frame_id, Object.class_id, Object.state, Object.source,
               Object.rig_object_id, ObjectEmbedding.dino_vec)
        .join(ObjectEmbedding, ObjectEmbedding.object_id == Object.object_id, isouter=True)
        .where(Object.frame_id.in_(list(cam_of_frame)), Object.state != "rejected"))).all()
    return [{"oid": oid, "cam": cam_of_frame[fid], "frame_id": fid, "class_id": cid, "state": st,
             "source": src, "rig_object_id": rid,
             "vec": np.asarray(v, np.float32) if v is not None else None}
            for oid, fid, cid, st, src, rid, v in rows]


async def suggest_links(session_id: UUID, group_id: UUID, appearance_cos: float = 0.55) -> dict:
    """Propose cross-camera link candidates by DINOv3 appearance cosine (assist only, never applied). Objects
    already sharing a rig identity are skipped. Returns pairs sorted by similarity for one-click accept."""
    maker = get_sessionmaker()
    async with maker() as db:
        group = await db.get(FrameGroup, group_id)
        if group is None:
            return {"suggestions": [], "reason": "group not found"}
        objs = await _group_objects(db, group)
        pairs = []
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                a, b = objs[i], objs[j]
                if a["cam"] == b["cam"] or a["vec"] is None or b["vec"] is None:
                    continue
                if a["rig_object_id"] and a["rig_object_id"] == b["rig_object_id"]:
                    continue  # already linked together
                cos = float(a["vec"] @ b["vec"])
                if cos >= appearance_cos:
                    pairs.append({"a": str(a["oid"]), "b": str(b["oid"]), "cam_a": a["cam"], "cam_b": b["cam"],
                                  "class_a": a["class_id"], "class_b": b["class_id"], "cos": round(cos, 3)})
        pairs.sort(key=lambda p: p["cos"], reverse=True)
        return {"group_id": str(group_id), "suggestions": pairs, "appearance_cos": appearance_cos}


async def link_objects(session_id: UUID, group_id: UUID, object_ids: list[UUID], source: str = "manual") -> dict:
    """Bind objects into one rig identity, merging any rig objects they already belong to. Recomputes the voted
    class and conflict flag, stamps object.rig_object_id, and routes non-human members to review on conflict."""
    maker = get_sessionmaker()
    async with maker() as db:
        group = await db.get(FrameGroup, group_id)
        if group is None:
            return {"error": "group not found"}
        objs = {o["oid"]: o for o in await _group_objects(db, group)}
        targets = [oid for oid in object_ids if oid in objs]
        if len(targets) < 2:
            return {"error": "need at least two objects in this group to link"}

        # gather any existing rig objects among the targets to merge them into one
        existing_ids = {objs[oid]["rig_object_id"] for oid in targets if objs[oid]["rig_object_id"]}
        existing = []
        for rid in existing_ids:
            ro = await db.get(RigObject, rid)
            if ro is not None:
                existing.append(ro)

        member_ids: set[UUID] = set(targets)
        link_sources: dict[str, str] = {}
        for ro in existing:
            member_ids.update(ro.member_object_ids or [])
            link_sources.update(ro.link_sources or {})
        for oid in targets:
            link_sources[str(oid)] = source

        # recompute class vote across the final member set (fetch classes for members outside this group's map)
        classes = []
        for mid in member_ids:
            if mid in objs:
                classes.append(objs[mid]["class_id"])
            else:
                o = await db.get(Object, mid)
                if o is not None:
                    classes.append(o.class_id)
        voted, conflict = _vote(classes)

        keep = existing[0] if existing else None
        if keep is None:
            keep = RigObject(session_id=session_id, group_id=group_id)
            db.add(keep)
            await db.flush()
        keep.member_object_ids = sorted(member_ids, key=str)
        keep.link_sources = link_sources
        keep.class_id = voted
        keep.conflict = conflict
        keep.provenance = {"source": source}

        # delete the other merged rig objects, then stamp members and route conflicts to review
        for ro in existing[1:]:
            await db.delete(ro)
        await db.execute(update(Object).where(Object.object_id.in_(list(member_ids)))
                         .values(rig_object_id=keep.rig_object_id))
        if conflict:
            # a cross-view class disagreement: send the non-human members back to the review queue
            await db.execute(update(Object)
                             .where(Object.object_id.in_(list(member_ids)), Object.source != "human")
                             .values(state="review"))
        await db.commit()
        rid = keep.rig_object_id
        n = len(member_ids)
    log.info("multicam.linked", session_id=str(session_id), rig_object_id=str(rid), members=n, conflict=conflict)
    return {"rig_object_id": str(rid), "members": n, "class_id": voted, "conflict": conflict}


async def unlink_object(object_id: UUID) -> dict:
    """Remove an object from its rig identity. If fewer than two members remain, dissolve the rig object."""
    maker = get_sessionmaker()
    async with maker() as db:
        o = await db.get(Object, object_id)
        if o is None or o.rig_object_id is None:
            return {"error": "object is not linked"}
        ro = await db.get(RigObject, o.rig_object_id)
        o.rig_object_id = None
        if ro is not None:
            members = [m for m in (ro.member_object_ids or []) if m != object_id]
            (ro.link_sources or {}).pop(str(object_id), None)
            if len(members) < 2:
                # dissolve: clear the remaining member and drop the rig object
                await db.execute(update(Object).where(Object.object_id.in_(members)).values(rig_object_id=None))
                await db.delete(ro)
                dissolved = True
            else:
                ro.member_object_ids = members
                classes = []
                for mid in members:
                    m = await db.get(Object, mid)
                    if m is not None:
                        classes.append(m.class_id)
                ro.class_id, ro.conflict = _vote(classes)
                dissolved = False
        await db.commit()
    return {"unlinked": str(object_id), "dissolved": dissolved if ro is not None else False}


async def rig_objects(session_id: UUID, group_id: UUID) -> dict:
    """The rig-first object list for a group: linked identities (members grouped by camera, voted class,
    conflict) followed by the still-unlinked singletons, so the reviewer sees one entry per physical object."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()

    def name(cid):
        try:
            return onto.by_id(int(cid)).name
        except Exception:  # noqa: BLE001
            return str(cid)

    maker = get_sessionmaker()
    async with maker() as db:
        group = await db.get(FrameGroup, group_id)
        if group is None:
            return {"rig_objects": [], "singletons": []}
        objs = await _group_objects(db, group)
        by_rig: dict[UUID, list[dict]] = defaultdict(list)
        singles = []
        for o in objs:
            if o["rig_object_id"]:
                by_rig[o["rig_object_id"]].append(o)
            else:
                singles.append(o)

        rigs = []
        for rid, members in by_rig.items():
            ro = await db.get(RigObject, rid)
            member_out = [{"object_id": str(m["oid"]), "cam": m["cam"], "class_id": m["class_id"],
                           "class_name": name(m["class_id"]), "state": m["state"]} for m in members]
            rigs.append({"rig_object_id": str(rid), "class_id": ro.class_id if ro else None,
                         "class_name": name(ro.class_id) if ro and ro.class_id is not None else None,
                         "conflict": bool(ro.conflict) if ro else False,
                         "cameras": sorted({m["cam"] for m in members}), "members": member_out})
        rigs.sort(key=lambda r: (not r["conflict"], r["class_name"] or ""))
        singletons = [{"object_id": str(s["oid"]), "cam": s["cam"], "class_id": s["class_id"],
                       "class_name": name(s["class_id"]), "state": s["state"]} for s in singles]
        singletons.sort(key=lambda s: (s["cam"], s["class_name"]))
        return {"group_id": str(group_id), "rig_objects": rigs, "singletons": singletons}
