"""Recall recovery endpoints (optional). Trigger a recall run over a session (reviewer or admin, deny by
default), and list recall candidates by status. Recovered objects also surface in the existing review
queue with no UI change, since they are ordinary review-state objects.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import RecallCandidate
from services.api.deps import db_session, require_role
from services.recall.recover import run_recall

router = APIRouter()


@router.post("/recall/run/{session_id}", dependencies=[Depends(require_role("reviewer"))])
async def run(session_id: str, shortlist_only: bool = False, db: AsyncSession = Depends(db_session)):
    """Source recall candidates over a session. With shortlist_only the model channels touch only the
    active-learning shortlist; trackgap always runs full-session."""
    frame_ids = None
    if shortlist_only:
        from services.activelearn.selector import score_candidates

        items = await score_candidates(db, session_id=session_id)
        frame_ids = sorted({it["frame_id"] for it in items})
    return await run_recall(db, session_id, frame_ids=frame_ids)


@router.get("/recall/candidates")
async def candidates(status: str = "pending", session_id: str | None = None, limit: int = 200,
                     db: AsyncSession = Depends(db_session)):
    q = select(RecallCandidate).where(RecallCandidate.status == status)
    if session_id:
        from db.models import Frame

        q = q.join(Frame, Frame.frame_id == RecallCandidate.frame_id).where(Frame.session_id == session_id)
    rows = (await db.execute(q.order_by(RecallCandidate.fn_value.desc()).limit(limit))).scalars().all()
    return {"status": status, "count": len(rows), "candidates": [
        {"candidate_id": str(c.candidate_id), "object_id": str(c.object_id), "frame_id": str(c.frame_id),
         "channels": c.channels, "fn_value": c.fn_value, "class_id": c.class_id, "status": c.status}
        for c in rows]}
