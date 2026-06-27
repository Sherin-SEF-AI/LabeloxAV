"""Human review actions. Every accept or correction writes a review row (the audit trail and the
active-learning training signal) and updates the object with source=human."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.timebase import now_ns
from db.models import Object, Review, TrainingJob
from services.api.deps import BulkReviewIn, ReviewIn, current_user, db_session
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
async def review_object(object_id: str, payload: ReviewIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    obj = await db.get(Object, UUID(object_id))
    if obj is None:
        raise HTTPException(404, "object not found")
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
        errors = onto.validate_attrs(payload.attrs)
        if errors:
            raise HTTPException(400, {"attr_errors": errors})
        merged = dict(obj.attrs or {})
        merged.update(payload.attrs)
        obj.attrs = merged

    # State: explicit override wins, else derive from the action verb.
    obj.state = payload.state or _ACTION_STATE.get(payload.action, obj.state)
    obj.source = "human"

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
    }
