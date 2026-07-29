"""Named saves of a frame's annotations: save as, list, restore, delete.

Restore is the only destructive route here and it is a POST that returns the id of the checkpoint it took of
what it replaced, so the client can offer to undo it immediately rather than telling somebody their hour is
gone.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import current_user, db_session, require_role

router = APIRouter()


class CheckpointIn(BaseModel):
    """A save-as. The name is what a person will scan a list for, so it is required and the note is not."""

    name: str
    note: str | None = None


@router.post("/frames/{frame_id}/checkpoints", dependencies=[Depends(require_role("annotator"))])
async def save_as(frame_id: UUID, body: CheckpointIn, db: AsyncSession = Depends(db_session),
                  user=Depends(current_user)):
    """Save the frame's annotations as they stand, under a name."""
    from services.annotate.checkpoints import create_checkpoint

    if not body.name.strip():
        raise HTTPException(400, "a checkpoint needs a name, or nobody can find it again")
    try:
        return await create_checkpoint(db, frame_id, name=body.name, note=body.note,
                                       created_by=getattr(user, "name", None))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/frames/{frame_id}/checkpoints")
async def list_for_frame(frame_id: UUID, include_auto: bool = True,
                         limit: int = Query(50, ge=1, le=200),
                         db: AsyncSession = Depends(db_session)):
    """This frame's saves, newest first.

    The automatic ones a restore left behind are included and flagged rather than hidden, because they are
    the states restores replaced and that is exactly what somebody hunting a lost state wants.
    """
    from services.annotate.checkpoints import list_checkpoints

    return await list_checkpoints(db, frame_id, include_auto=include_auto, limit=limit)


@router.get("/checkpoints/{checkpoint_id}")
async def get_one(checkpoint_id: UUID, db: AsyncSession = Depends(db_session)):
    """One checkpoint including its objects, for previewing before restoring."""
    from services.annotate.checkpoints import get_checkpoint

    got = await get_checkpoint(db, checkpoint_id)
    if got is None:
        raise HTTPException(404, "checkpoint not found")
    return got


@router.post("/checkpoints/{checkpoint_id}/restore",
             dependencies=[Depends(require_role("annotator"))])
async def restore(checkpoint_id: UUID, db: AsyncSession = Depends(db_session),
                  user=Depends(current_user)):
    """Put the frame back to this state.

    Takes a checkpoint of the present first and returns its id as `undo_with`, so a restore triggered by
    mistake is one call from being reversed.
    """
    from services.annotate.checkpoints import restore_checkpoint

    try:
        return await restore_checkpoint(db, checkpoint_id, created_by=getattr(user, "name", None))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.delete("/checkpoints/{checkpoint_id}", dependencies=[Depends(require_role("annotator"))])
async def remove(checkpoint_id: UUID, db: AsyncSession = Depends(db_session)):
    from services.annotate.checkpoints import delete_checkpoint

    if not await delete_checkpoint(db, checkpoint_id):
        raise HTTPException(404, "checkpoint not found")
    return {"deleted": str(checkpoint_id)}


@router.get("/objects/{object_id}/history")
async def object_history(object_id: UUID, db: AsyncSession = Depends(db_session)):
    """Everything that has happened to one object, in order.

    The audit trail already recorded this and nothing ever showed it. A reviewer asking "why is this a bus"
    had to take the current state on faith; now the answer is a list of who changed what, when, and what it
    was before.
    """
    from sqlalchemy import select

    from db.models import Object, Review

    obj = await db.get(Object, object_id)
    if obj is None:
        raise HTTPException(404, "object not found")

    rows = (await db.execute(
        select(Review).where(Review.object_id == object_id)
        .order_by(Review.ts_ns))).scalars().all()

    return {
        "object_id": str(object_id),
        "current": {"class_id": obj.class_id, "state": obj.state, "version": obj.version,
                    "source": obj.source, "conf": obj.conf},
        "count": len(rows),
        "events": [{"review_id": str(r.review_id), "action": r.action, "reviewer": r.reviewer,
                    "ts_ns": r.ts_ns, "before": r.before, "after": r.after,
                    "time_spent_ms": r.time_spent_ms} for r in rows],
        "reasoning": (obj.provenance or {}).get("reasoning"),
    }
