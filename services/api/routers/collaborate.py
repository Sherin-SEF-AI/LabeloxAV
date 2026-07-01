"""Collaboration endpoints (M4.3): assignments, isolated annotator branches, merge requests with reviewed
merge + revert, and lakeFS branch listing."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session
from versioning import collaborate as C
from versioning import lakefs_store as L

router = APIRouter()


@router.get("/collaborate/branches")
async def branches():
    return {"branches": L.list_branches()}


@router.get("/collaborate/assignments")
async def assignments(user_id: str | None = None, db: AsyncSession = Depends(db_session)):
    return await C.list_assignments(db, user_id)


@router.get("/collaborate/merge_requests")
async def merge_requests(db: AsyncSession = Depends(db_session)):
    return await C.list_merge_requests(db)


class AssignIn(BaseModel):
    item_id: str
    user_id: str
    branch: str | None = None


@router.post("/collaborate/assign")
async def assign(payload: AssignIn, db: AsyncSession = Depends(db_session)):
    return await C.create_assignment(db, payload.item_id, payload.user_id, payload.branch)


class EnqueueIn(BaseModel):
    item_ids: list
    user_id: str


@router.post("/collaborate/tasks/enqueue")
async def enqueue_tasks_ep(payload: EnqueueIn):
    """Milestone I: bulk-enqueue a work list (slice members or ranked candidates) as tasks for a user."""
    from uuid import UUID

    from services.tasks.queue import enqueue_tasks
    return await enqueue_tasks(payload.item_ids, UUID(payload.user_id))


@router.post("/collaborate/tasks/claim")
async def claim_task_ep(user_id: str):
    """Milestone I: atomically claim the next pending task (race-safe; two annotators never get the same)."""
    from uuid import UUID

    from services.tasks.queue import claim_next
    return await claim_next(UUID(user_id))


@router.post("/collaborate/tasks/{assignment_id}/advance")
async def advance_task_ep(assignment_id: str, to_status: str):
    """Milestone I: move a task through the queue state machine (refuses an invalid transition)."""
    from uuid import UUID

    from services.tasks.queue import advance_task
    res = await advance_task(UUID(assignment_id), to_status)
    if res.get("error"):
        raise HTTPException(404, res["error"])
    if res.get("invalid_transition"):
        raise HTTPException(409, res)
    return res


@router.get("/collaborate/tasks/stats")
async def task_stats_ep(user_id: str):
    """Milestone I: task counts per status for a user (queue depth + work in progress)."""
    from uuid import UUID

    from services.tasks.queue import queue_stats
    return await queue_stats(UUID(user_id))


class CommitWorkIn(BaseModel):
    labels: dict[str, dict]
    message: str = "annotator work"


@router.post("/collaborate/assignments/{assignment_id}/commit")
async def commit_work(assignment_id: str, payload: CommitWorkIn, db: AsyncSession = Depends(db_session)):
    return await C.commit_assignment_work(db, assignment_id, payload.labels, payload.message)


class OpenMRIn(BaseModel):
    title: str
    source_branch: str
    author_id: str | None = None
    notes: str | None = None


@router.post("/collaborate/merge_requests/open")
async def open_mr(payload: OpenMRIn, db: AsyncSession = Depends(db_session)):
    return await C.open_merge_request(db, payload.title, payload.source_branch, payload.author_id, notes=payload.notes)


class ReviewerIn(BaseModel):
    reviewer_id: str


@router.post("/collaborate/merge_requests/{mr_id}/approve")
async def approve_mr(mr_id: str, payload: ReviewerIn, db: AsyncSession = Depends(db_session)):
    return await C.approve_merge_request(db, mr_id, payload.reviewer_id)


@router.post("/collaborate/merge_requests/{mr_id}/merge")
async def merge_mr(mr_id: str, payload: ReviewerIn, db: AsyncSession = Depends(db_session)):
    return await C.merge_request(db, mr_id, payload.reviewer_id)


@router.post("/collaborate/merge_requests/{mr_id}/revert")
async def revert_mr(mr_id: str, payload: ReviewerIn, db: AsyncSession = Depends(db_session)):
    return await C.revert_merge_request(db, mr_id, payload.reviewer_id)
