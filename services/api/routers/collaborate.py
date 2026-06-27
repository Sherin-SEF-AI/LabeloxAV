"""Collaboration endpoints (M4.3): assignments, isolated annotator branches, merge requests with reviewed
merge + revert, and lakeFS branch listing."""

from __future__ import annotations

from fastapi import APIRouter, Depends
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
