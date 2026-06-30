"""Human review actions. Every accept or correction writes a review row (the audit trail and the
active-learning training signal) and updates the object with source=human."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from core.timebase import now_ns
from db.models import Frame, Object, Review, TrainingJob
from services.api.deps import BulkReviewIn, ReviewIn, current_user, db_session
from services.api.routers.objects import _write_mask
from services.autolabel.ontology import get_ontology

log = get_logger("api_review")
router = APIRouter()


@router.post("/qa/vlm")
async def qa_vlm(session_id: str, limit: int = 40, db: AsyncSession = Depends(db_session)):
    """Run a VLM auto-QA + auto-attributes pass on a session in the background: flags cross-superclass
    disagreements into the QA queue and pre-fills typed attributes. GPU-light (Ollama), yields to
    training. Flagged objects surface in triage's QA queue (state=submitted)."""
    import asyncio
    from uuid import UUID as _UUID

    from sqlalchemy import select

    if (await db.execute(select(TrainingJob.job_id).where(TrainingJob.status == "running").limit(1))).first():
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
    from uuid import UUID as _UUID

    for oid in payload.object_ids:
        obj = await db.get(Object, _UUID(oid))
        if obj is None:
            continue
        before = {"class_id": obj.class_id, "bbox": list(obj.bbox), "attrs": dict(obj.attrs or {}), "state": obj.state}
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
        db.add(Review(object_id=obj.object_id, reviewer=reviewer, user_id=uid, action=payload.action,
                      before=before, after={"class_id": obj.class_id, "bbox": list(obj.bbox), "attrs": dict(obj.attrs or {}), "state": obj.state},
                      time_spent_ms=0, ts_ns=now_ns()))
        n += 1
    await db.commit()
    return {"updated": n, "action": payload.action}


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
