"""Human review actions. Every accept or correction writes a review row (the audit trail and the
active-learning training signal) and updates the object with source=human."""

from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.observability import spawn
from core.storage import get_object_store
from core.timebase import now_ns
from db.models import Frame, Object, Review
from services.api.deps import BulkReviewIn, ReviewIn, current_user, db_session
from services.api.routers.objects import _write_mask
from services.autolabel.ontology import get_ontology
from services.review_apply import AttrRejected, apply_review_batch
from services.review_batch import record_batch
from services.review_policy import ReviewStateError, state_for
from services.training.gpu_lease import training_holds_gpu

log = get_logger("api_review")
router = APIRouter()


@router.post("/qa/vlm")
async def qa_vlm(session_id: str, limit: int = 40, db: AsyncSession = Depends(db_session)):
    """Run a VLM auto-QA + auto-attributes pass on a session in the background: flags cross-superclass
    disagreements into the QA queue and pre-fills typed attributes. GPU-light (Ollama), yields to
    training. Flagged objects surface in triage's QA queue (state=submitted)."""
    from uuid import UUID as _UUID


    if await training_holds_gpu(db):
        raise HTTPException(503, "GPU reserved for a training job; VLM auto-QA is paused until it finishes")

    async def _run() -> None:
        from services.intelligence.vlm_qa import vlm_qa_session

        try:
            await vlm_qa_session(_UUID(session_id), limit)
        except Exception as exc:  # noqa: BLE001
            log.error("qa_vlm.failed", error=str(exc))

    spawn(_run(), name="_run")
    return {"started": True, "session_id": session_id, "limit": limit}



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
    try:
        new_state = state_for(payload.action, payload.state, getattr(user, "role", None), None)
    except ReviewStateError as exc:
        raise HTTPException(400, str(exc)) from exc
    reviewer, uid = _attrib(user, payload.reviewer)

    missing: list[str] = []
    from uuid import UUID as _UUID

    ids = payload.object_ids
    objects: list[Object] = []
    for oid in ids:
        obj = await db.get(Object, _UUID(oid))
        if obj is None:
            missing.append(oid)
            continue
        objects.append(obj)

    # The per-object loop lives in services/review_apply.py so the track relabel path gets the same lock,
    # the same role clamp and the same undo instead of a fourth copy of them. Bulk keeps its own behaviour
    # exactly: it does not skip human-sourced rows (the caller ticked them) and it does not guard class
    # moves (the correction dialog exists to span classes).
    try:
        res = await apply_review_batch(
            db, objects, action=payload.action, onto=onto, class_id=cid, attrs=payload.attrs,
            requested_state=payload.state, role=getattr(user, "role", None), source="human",
            reviewer=reviewer, uid=uid, expected_versions=payload.expected_versions,
            time_spent_ms=payload.time_spent_ms)
    except AttrRejected as exc:
        raise HTTPException(400, {"attr_errors": exc.errors, "object_id": exc.object_id}) from exc
    n, stale, batch_changes = res.n, res.stale, res.changes

    # One reversible unit for the whole batch, stamped in the SAME transaction as the edits it describes.
    #
    # This used to commit the edits first and stamp afterwards, so that a failed edit could not leave a run
    # claiming objects it never changed. That reasoning is right and the ordering was the wrong way to get
    # it: dying between the two commits left sixty objects changed and no run id on them, and revert_batch
    # keys ownership on that stamp - so the batch was silently un-revertible, which is the one thing this
    # machinery exists to prevent. In one transaction a failed edit takes the stamp down with it, and
    # anything that commits is always revertible.
    run_id = await record_batch(db, batch_changes, created_by=reviewer, commit=False,
                                policy={"action": payload.action, "class_name": payload.class_name,
                                        "attrs": payload.attrs or {}}) if batch_changes else None
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
            "skipped_missing": missing, "skipped_stale": stale,
            # The handle for taking the whole batch back in one move.
            "run_id": run_id}


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
        obj.attrs = onto.derive_attrs(merged, obj.class_id)

    if payload.rot_deg is not None:
        obj.rot_deg = payload.rot_deg
    if payload.keypoints is not None:
        obj.keypoints = payload.keypoints
    if payload.polyline is not None:
        obj.polyline = payload.polyline
    if payload.cuboid_3d is not None:
        obj.cuboid_3d = payload.cuboid_3d
    if payload.mask_polygons is not None:
        # Geometry and mask are written in one request rather than through a separate updateMask call, so
        # a client cannot leave them out of sync by making only one of two calls.
        #
        # Not a transaction, and the comment here used to say it was. Object storage does not enlist in the
        # database's transaction and cannot be made to, so if the commit below fails the blob has already
        # been written. What makes that harmless is the key: it is deterministic in (session, frame,
        # object), so nothing references the orphan - obj.mask_uri was rolled back with everything else -
        # and the next successful save of this object overwrites it in place. The failure mode is a stale
        # blob at a key that will be reused, not a mask attached to the wrong object.
        frame = await db.get(Frame, obj.frame_id)
        # The PUT is a network round trip; it was running on the event loop.
        mask_uri = await asyncio.to_thread(
            _write_mask, get_object_store(), frame.session_id, frame.frame_id, obj.object_id,
            payload.mask_polygons, frame.width, frame.height)
        obj.mask_uri = mask_uri
        obj.mask_encoding = "polygon"

    # The state depends on the caller's role, not only on the verb. An annotator's accept is a submission,
    # which is the QA queue the triage page's `submitted` band exists to serve. This used to take the state
    # straight from the request body, so the whole review step was advisory.
    try:
        new_state = state_for(payload.action, payload.state, getattr(user, "role", None), obj.state)
    except ReviewStateError as exc:
        raise HTTPException(400, str(exc)) from exc
    if new_state is not None:
        obj.state = new_state
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
