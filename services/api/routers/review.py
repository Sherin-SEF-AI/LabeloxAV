"""Human review actions. Every accept or correction writes a review row (the audit trail and the
active-learning training signal) and updates the object with source=human."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from core.timebase import now_ns
from db.models import Frame, Object, Review
from services.api.deps import BulkReviewIn, ReviewIn, current_user, db_session
from services.api.routers.objects import _write_mask
from services.autolabel.ontology import get_ontology
from services.training.gpu_lease import training_holds_gpu

log = get_logger("api_review")
router = APIRouter()


@router.post("/qa/vlm")
async def qa_vlm(session_id: str, limit: int = 40, db: AsyncSession = Depends(db_session)):
    """Run a VLM auto-QA + auto-attributes pass on a session in the background: flags cross-superclass
    disagreements into the QA queue and pre-fills typed attributes. GPU-light (Ollama), yields to
    training. Flagged objects surface in triage's QA queue (state=submitted)."""
    import asyncio
    from uuid import UUID as _UUID


    if await training_holds_gpu(db):
        raise HTTPException(503, "GPU reserved for a training job; VLM auto-QA is paused until it finishes")

    async def _run() -> None:
        from services.intelligence.vlm_qa import vlm_qa_session

        try:
            await vlm_qa_session(_UUID(session_id), limit)
        except Exception as exc:  # noqa: BLE001
            log.error("qa_vlm.failed", error=str(exc))

    asyncio.create_task(_run())
    return {"started": True, "session_id": session_id, "limit": limit}

_ACTION_STATE = {"confirm": "accepted", "accept": "accepted", "reject": "rejected"}


def _attrib(user, fallback: str) -> tuple[str, object]:
    """Return (reviewer_name, user_id) for the acting user, falling back to a payload name."""
    return (user.name, user.user_id) if user is not None else (fallback, None)


@router.post("/objects/bulk-review")
async def bulk_review(payload: BulkReviewIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Apply one review action to many objects at once (accept/reject/relabel a filtered set). Each
    object gets source=human + a Review audit row, exactly like single review."""
    onto = get_ontology()
    cid = None
    if payload.class_name is not None:
        if not onto.has_name(payload.class_name):
            raise HTTPException(400, f"unknown class '{payload.class_name}'")
        cid = onto.by_name(payload.class_name).id
    new_state = payload.state or _ACTION_STATE.get(payload.action)
    reviewer, uid = _attrib(user, payload.reviewer)

    n = 0
    missing: list[str] = []
    stale: list[dict] = []
    from uuid import UUID as _UUID

    expected = payload.expected_versions or {}
    # The batch's time is divided across its members rather than attributed to any one of them, so a grid
    # that triages sixty crops in a minute reports sixty seconds of work and not sixty minutes or zero.
    ids = payload.object_ids
    per_item_ms = int(payload.time_spent_ms / len(ids)) if ids and payload.time_spent_ms > 0 else 0

    for oid in ids:
        obj = await db.get(Object, _UUID(oid))
        if obj is None:
            missing.append(oid)
            continue
        # Optimistic lock, per object. Single review has refused a stale write since it was written; bulk
        # review did not, so a sweep silently overwrote whatever anybody else had done in the meantime. A
        # stale member is skipped and named rather than failing the whole batch, because one contended
        # object should not discard fifty-nine good verdicts.
        want = expected.get(oid)
        if want is not None and obj.version != want:
            stale.append({"object_id": oid, "expected": want, "current": obj.version})
            continue
        before = {"class_id": obj.class_id, "bbox": list(obj.bbox), "attrs": dict(obj.attrs or {}), "state": obj.state,
                  "source": obj.source, "conf": obj.conf, "provenance": dict(obj.provenance or {})}
        if cid is not None:
            obj.class_id = cid
        if payload.attrs:
            errors = onto.validate_attrs(payload.attrs, obj.class_id)   # against the effective (possibly new) class
            if errors:
                raise HTTPException(400, {"attr_errors": errors, "object_id": oid})
            merged = dict(obj.attrs or {})
            merged.update(payload.attrs)
            obj.attrs = merged
        if new_state is not None:
            obj.state = new_state
        obj.source = "human"
        # Advance the lock version, exactly as single review does. Without this a bulk edit was invisible to
        # every other client's optimistic check, so an editor holding the object would overwrite it back.
        obj.version = (obj.version or 1) + 1
        db.add(Review(object_id=obj.object_id, reviewer=reviewer, user_id=uid, action=payload.action,
                      before=before, after={"class_id": obj.class_id, "bbox": list(obj.bbox), "attrs": dict(obj.attrs or {}), "state": obj.state},
                      time_spent_ms=per_item_ms, ts_ns=now_ns()))
        n += 1
    await db.commit()

    # One activity entry for the batch, not one per object: the feed is a human timeline and sixty identical
    # rows is not a record of what somebody did, it is noise that buries everything either side of it.
    if n:
        from services.activity import record_activity

        _verbs = {"confirm": "confirmed", "accept": "confirmed", "reject": "rejected",
                  "reclassify": "reclassified"}
        await record_activity(
            db, user_id=uid, user_name=reviewer,
            verb=_verbs.get(payload.action, "reviewed"),
            subject_type="object", subject_id=str(ids[0]),
            summary=(f"{payload.action} {n} objects"
                     + (f" as {payload.class_name}" if payload.class_name else "")),
            href=f"/review/grid?ids={n}",
            meta={"action": payload.action, "count": n, "state": new_state,
                  "skipped_stale": len(stale), "skipped_missing": len(missing)})

    return {"updated": n, "action": payload.action,
            # Named rather than silently dropped: a caller that asked for sixty and changed fifty-eight has
            # to be able to find out which two, and why.
            "skipped_missing": missing, "skipped_stale": stale}


@router.post("/objects/{object_id}/review")
async def review_object(object_id: UUID, payload: ReviewIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    obj = await db.get(Object, object_id)
    if obj is None:
        raise HTTPException(404, "object not found")
    # Optimistic lock: if the editor's view is stale (someone else edited since), refuse with 409 so the
    # client can reload rather than clobber the other annotator's change.
    if payload.expected_version is not None and obj.version != payload.expected_version:
        raise HTTPException(409, {"detail": "object changed since you loaded it", "current_version": obj.version})
    onto = get_ontology()
    reviewer, uid = _attrib(user, payload.reviewer)

    before = {
        "class_id": obj.class_id,
        "bbox": list(obj.bbox),
        "attrs": dict(obj.attrs or {}),
        "state": obj.state,
        # Capture the prediction provenance the review is about to overwrite. Without source/conf here the
        # original detection is unrecoverable, and the eval harness can never reconstruct a PR curve for it.
        "source": obj.source,
        "conf": obj.conf,
        "provenance": dict(obj.provenance or {}),
    }

    if payload.class_name is not None:
        if not onto.has_name(payload.class_name):
            raise HTTPException(400, f"unknown class '{payload.class_name}'")
        obj.class_id = onto.by_name(payload.class_name).id

    if payload.bbox is not None:
        if len(payload.bbox) != 4:
            raise HTTPException(400, "bbox must be [x1,y1,x2,y2]")
        obj.bbox = payload.bbox

    if payload.attrs is not None:
        errors = onto.validate_attrs(payload.attrs, obj.class_id)
        if errors:
            raise HTTPException(400, {"attr_errors": errors})
        merged = dict(obj.attrs or {})
        merged.update(payload.attrs)
        obj.attrs = merged

    if payload.rot_deg is not None:
        obj.rot_deg = payload.rot_deg
    if payload.keypoints is not None:
        obj.keypoints = payload.keypoints
    if payload.polyline is not None:
        obj.polyline = payload.polyline
    if payload.cuboid_3d is not None:
        obj.cuboid_3d = payload.cuboid_3d
    if payload.mask_polygons is not None:
        # Write the mask blob in the same request so geometry + mask persist atomically (one transaction),
        # instead of a separate updateMask call that can leave them out of sync on a partial failure.
        frame = await db.get(Frame, obj.frame_id)
        obj.mask_uri = _write_mask(get_object_store(), frame.session_id, frame.frame_id, obj.object_id,
                                   payload.mask_polygons, frame.width, frame.height)
        obj.mask_encoding = "polygon"

    # State: explicit override wins, else derive from the action verb.
    obj.state = payload.state or _ACTION_STATE.get(payload.action, obj.state)
    obj.source = "human"
    obj.version = (obj.version or 1) + 1  # advance the optimistic-lock version on every human edit

    after = {
        "class_id": obj.class_id,
        "bbox": list(obj.bbox),
        "attrs": dict(obj.attrs or {}),
        "state": obj.state,
    }
    db.add(
        Review(
            object_id=obj.object_id,
            reviewer=reviewer,
            user_id=uid,
            action=payload.action,
            before=before,
            after=after,
            time_spent_ms=payload.time_spent_ms,
            ts_ns=now_ns(),
        )
    )
    await db.commit()

    # The activity feed. Recorded here rather than derived from the review table later because the feed is
    # one timeline across reviews, drawings, jobs, and exports, and reconstructing that by unioning five
    # tables at read time is what made "what did I do today" unanswerable.
    from services.activity import record_activity

    _verbs = {"confirm": "confirmed", "reject": "rejected", "create": "created_object"}
    await record_activity(
        db, user_id=uid, user_name=reviewer,
        verb=_verbs.get(payload.action, "reviewed"),
        subject_type="object", subject_id=str(obj.object_id),
        summary=f"{payload.action} {onto.by_id(obj.class_id).name}",
        href=f"/frame/{obj.frame_id}",
        meta={"action": payload.action, "state": obj.state})

    return {
        "object_id": str(obj.object_id),
        "class_id": obj.class_id,
        "class_name": onto.by_id(obj.class_id).name,
        "bbox": list(obj.bbox),
        "attrs": obj.attrs,
        "state": obj.state,
        "source": obj.source,
        "version": obj.version,
        "rot_deg": obj.rot_deg or 0.0,
        "keypoints": obj.keypoints,
    }
