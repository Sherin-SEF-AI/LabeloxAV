"""Ontology and session listing endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame, Object, OntologyClass, Review
from db.models import Session as DbSession
from platforms.registry import as_dicts as platform_dicts
from services.api.deps import OntologyClassOut, current_user, db_session, require_role
from services.autolabel.ontology import add_custom_class, get_ontology

router = APIRouter()


@router.get("/platforms")
async def list_platforms():
    """The platform registry: the seven platforms the data engine hosts (annotation core + the six folded
    subsystems), in launcher order. The frontend mirror (web/platforms/registry.ts) must match these ids;
    a test reconciles the two."""
    return {"platforms": platform_dicts()}


class NewClassIn(BaseModel):
    name: str
    l0: str = "object"
    l1: str = "custom"
    india: bool = True


@router.post("/ontology/classes", dependencies=[Depends(require_role("reviewer"))])
async def create_class(payload: NewClassIn, db: AsyncSession = Depends(db_session)):
    """Add an annotator-defined custom class. It lands in the custom id block, is marked rare so the gate
    routes it to human review, and is mirrored into the DB ontology table for the current version.

    Reviewer-gated. Minting a class is not a per-object edit: it changes the vocabulary every subsequent
    label is drawn from, and a class the sidecar offers but the database will not store is what killed a
    corpus relabel run at frame 13 of 25.
    """
    try:
        cls = add_custom_class(payload.name, payload.l0, payload.l1, payload.india)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    onto = get_ontology()
    if await db.get(OntologyClass, cls["id"]) is None:
        db.add(OntologyClass(id=cls["id"], version=onto.version, name=cls["name"], l0=cls["l0"],
                             l1=cls["l1"], india=cls["india"], map_to={}))
        await db.commit()
    return cls


class MergeIn(BaseModel):
    from_id: int
    to_id: int


class RenameIn(BaseModel):
    class_id: int
    new_name: str


@router.post("/ontology/classes/merge", dependencies=[Depends(require_role("admin"))])
async def merge_classes(payload: MergeIn, user=Depends(current_user),
                        db: AsyncSession = Depends(db_session)):
    """Move every object and track from one class into another, as one reversible run.

    `merge_class` has existed in services/agent/ontology_merge.py and been reachable from nothing: the only
    ontology write in the application was minting a class, so a mistake in the vocabulary could be added to
    and never repaired. Admin-gated because it rewrites the class of every object that carries it.
    """
    from services.agent.ontology_merge import MergeError, merge_class

    try:
        return await merge_class(db, from_id=payload.from_id, to_id=payload.to_id,
                                 created_by=str(user.user_id) if user else None)
    except MergeError as exc:
        raise HTTPException(400, str(exc)) from None


@router.post("/ontology/merges/{run_id}/revert", dependencies=[Depends(require_role("admin"))])
async def revert_merge_run(run_id: str, db: AsyncSession = Depends(db_session)):
    """Undo a merge, putting every object back on the class it carried."""
    import uuid as _uuid

    from db.models import AgentRun
    from services.agent.ontology_merge import KIND, revert_merge

    run = await db.get(AgentRun, _uuid.UUID(run_id))
    if run is None or run.kind != KIND:
        raise HTTPException(404, "no such merge run")
    return await revert_merge(db, run)


@router.post("/ontology/classes/rename", dependencies=[Depends(require_role("admin"))])
async def rename_class(payload: RenameIn, db: AsyncSession = Depends(db_session)):
    """Rename a custom class, keeping its id so every object that carries it follows the rename.

    Only sidecar classes: the governed YAML is the versioned vocabulary and is edited in the file, not
    through an endpoint that would leave the file and the database disagreeing.
    """
    from services.agent.ontology_merge import MergeError, rename_in_sidecar

    try:
        out = rename_in_sidecar(payload.class_id, payload.new_name)
    except MergeError as exc:
        raise HTTPException(400, str(exc)) from None
    # The DB mirror has to follow, or the API serves one name and every join serves the other.
    row = await db.get(OntologyClass, payload.class_id)
    if row is not None:
        row.name = out["to"]
        await db.commit()
    return out


@router.post("/ontology/classes/retire", dependencies=[Depends(require_role("admin"))])
async def retire_classes(class_ids: list[int], db: AsyncSession = Depends(db_session)):
    """Stop offering these classes, without deleting the rows objects still point at.

    Retiring removes a class from the sidecar so nothing new can be labelled with it. The database row
    stays, because `prediction` and `eval_patch` hold immutable history that still references it, and
    deleting it would either fail on a foreign key or take that history with it.
    """
    from services.agent.ontology_merge import retire_from_sidecar

    still_used = {
        cid: (await db.execute(
            select(func.count()).select_from(Object).where(Object.class_id == cid))).scalar()
        for cid in class_ids
    }
    in_use = {cid: n for cid, n in still_used.items() if n}
    if in_use:
        # Retiring a class objects still carry would leave them on a name nothing offers, which is exactly
        # how a corpus ends up with labels no picker can select and no reviewer can correct.
        raise HTTPException(400, {"detail": "these classes are still on objects; merge them first",
                                  "in_use": in_use})
    return retire_from_sidecar(set(class_ids))


@router.get("/ontology")
async def ontology():
    onto = get_ontology()
    return {
        "version": onto.version,
        "hierarchy_levels": onto.hierarchy_levels,
        "attributes": {
            n: {"type": a.type, "values": a.values, "range": list(a.range) if a.range else None}
            for n, a in onto.attributes.items()
        },
        "classes": [OntologyClassOut(id=c.id, name=c.name, l0=c.l0, l1=c.l1, india=c.india).model_dump()
                    for c in sorted(onto.classes, key=lambda c: c.id)],
        # Per-subclass (l1) applicable-attribute allowlist, so the editor shows only the relevant attributes
        # for an object's class. A subclass absent here means all attributes apply.
        "attribute_scope": onto.attribute_scope,
    }


_IMPORT_VEHICLES = {"MAPILLARY", "IDD", "BDD", "BDD100K", "KITTI", "NUSCENES", "IMPORT-01"}


def _session_origin(s) -> str:
    """Where a session's data came from: imported (public dataset), fleet (your own dashcam drives), or
    other (synthetic/test). Lets the UI make clear that a Mapillary frame is not your own annotation."""
    v = (s.vehicle_id or "").upper()
    r = s.raw_uri or ""
    if "import_staging" in r or v in _IMPORT_VEHICLES:
        return "imported"
    if v.startswith("DASHCAM"):
        return "fleet"
    return "other"


def _session_row(s) -> dict:
    return {
        "session_id": str(s.session_id),
        "vehicle_id": s.vehicle_id,
        "city": s.city,
        "route": s.route,
        "start_ts_ns": s.start_ts_ns,
        "end_ts_ns": s.end_ts_ns,
        "ontology_version": s.ontology_version,
        "origin": _session_origin(s),
    }


@router.get("/sessions")
async def sessions(db: AsyncSession = Depends(db_session), limit: int = Query(200, ge=1, le=1000)):
    rows = (await db.execute(select(DbSession).order_by(DbSession.created_at.desc()).limit(limit))).scalars().all()
    return [_session_row(s) for s in rows]


@router.get("/sessions/page")
async def sessions_page(db: AsyncSession = Depends(db_session),
                        limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                        vehicle_id: str | None = None):
    """Paginated session list with a total, so the browser can page through all 2000+ drives instead of
    being silently capped at the first window. Optional vehicle_id narrows to one fleet."""
    base = select(DbSession)
    if vehicle_id:
        base = base.where(DbSession.vehicle_id == vehicle_id)
    total = (await db.execute(select(func.count()).select_from(base.subquery()))).scalar_one()
    rows = (await db.execute(
        base.order_by(DbSession.created_at.desc()).offset(offset).limit(limit))).scalars().all()
    return {"total": int(total), "offset": offset, "limit": limit,
            "sessions": [_session_row(s) for s in rows]}


@router.get("/sessions/states")
async def session_states(db: AsyncSession = Depends(db_session)):
    """One row per session saying what state the work is in, for the editor's drive picker.

    The picker needs to separate a drive nobody has started from one somebody is part way through, and the
    existing progress fraction cannot: accepted-plus-auto_accept over total has a median of 0.011 across
    this corpus, so a percentage bar reads about 1% on nearly every drive and tells the annotator nothing.
    What does separate them is coarser and true: whether the frames carry any detections at all (42
    sessions have none), and whether a PERSON has ruled on any of it (125 have).

    `user_id IS NOT NULL` is what makes the second half honest. Machine writers put rows in `review` too:
    27,396 of them here are from the track-relabel backfill alone, against 1,613 written by a human, and
    counting those would mark almost every drive as reviewed by somebody who never opened it.

    Two queries rather than one join, because the human-review side is small and joining it against 714,415
    objects to count 1,613 rows is the expensive way to get the same answer.
    """
    # Driven from Session, not Frame. Driving it from Frame returned 252 rows against 377 sessions and
    # silently omitted the 126 that have no camera frames, which are exactly the ones this endpoint exists
    # to mark: they are LiDAR and 3D captures, and the editor 404s on them.
    frames_q = (await db.execute(
        select(DbSession.session_id, func.count(func.distinct(Frame.frame_id)), func.count(Object.object_id))
        .select_from(DbSession)
        .outerjoin(Frame, Frame.session_id == DbSession.session_id)
        .outerjoin(Object, Object.frame_id == Frame.frame_id)
        .group_by(DbSession.session_id))).all()

    reviewed_q = (await db.execute(
        select(Frame.session_id, func.count(func.distinct(Review.object_id)))
        .select_from(Review)
        .join(Object, Object.object_id == Review.object_id)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(Review.user_id.is_not(None))
        .group_by(Frame.session_id))).all()
    reviewed = {str(sid): int(n) for sid, n in reviewed_q}

    return [{"session_id": str(sid), "frames": int(fr), "objects": int(ob),
             "reviewed_objects": reviewed.get(str(sid), 0)}
            for sid, fr, ob in frames_q]


@router.get("/sessions/{session_id}/stats")
async def session_stats(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """Per-session progress for the Open Annotation browser: frame count + object counts by state, and a
    progress fraction (auto-accepted + human-accepted over total)."""
    frames = (await db.execute(
        select(func.count()).select_from(Frame).where(Frame.session_id == session_id))).scalar_one()
    rows = (await db.execute(
        select(Object.state, func.count()).join(Frame, Object.frame_id == Frame.frame_id)
        .where(Frame.session_id == session_id).group_by(Object.state))).all()
    by_state = {k: int(v) for k, v in rows}
    objects = sum(by_state.values())
    done = by_state.get("auto_accept", 0) + by_state.get("accepted", 0)
    return {"session_id": str(session_id), "frames": int(frames), "objects": objects,
            "by_state": by_state, "done": done,
            "progress": round(done / objects, 3) if objects else 0.0}


@router.get("/sessions/{session_id}/first-frame")
async def first_frame(session_id: uuid.UUID, db: AsyncSession = Depends(db_session)):
    """The chronological first frame of a session, so a card can open it directly in the editor."""
    fid = (await db.execute(select(Frame.frame_id).where(Frame.session_id == session_id)
           .order_by(Frame.ts_ns.asc()).limit(1))).scalar_one_or_none()
    if fid is None:
        raise HTTPException(404, "no frames in session")
    return {"frame_id": str(fid)}
