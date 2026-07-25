"""Labeling operations endpoints: projects, tasks, assignable jobs, issue threads, and annotator scorecards.

Role floors follow the existing convention: annotators work their own jobs and raise issues; creating
projects/tasks and assigning work is a reviewer action, since it directs other people's time.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import current_user, db_session, require_role
from services.labelops import issues as issue_svc
from services.labelops import jobs as job_svc
from services.labelops import quality as quality_svc

router = APIRouter()


def _uid(user) -> str | None:
    return str(user.user_id) if user is not None else None


class ProjectIn(BaseModel):
    name: str
    description: str | None = None
    modality: str = "image"
    honeypot_frac: float = 0.0
    min_honeypot_accuracy: float = 0.9
    gold_id: str | None = None


class TaskIn(BaseModel):
    project_id: str
    name: str
    session_id: str | None = None
    predicate: dict = {}
    jobs_of: int = 50


class AssignIn(BaseModel):
    assignee_id: str | None = None
    expected_version: int | None = None


class StateIn(BaseModel):
    state: str
    expected_version: int | None = None


class SubmitIn(BaseModel):
    expected_version: int | None = None


class IssueIn(BaseModel):
    kind: str = "comment"
    body: str | None = None
    object_id: str | None = None
    frame_id: str | None = None
    job_id: str | None = None
    region: list | None = None


class CommentIn(BaseModel):
    body: str


# ---- projects and tasks ---------------------------------------------------------------------------------

@router.post("/labelops/projects")
async def create_project(payload: ProjectIn, user=Depends(require_role("reviewer")),
                         db: AsyncSession = Depends(db_session)):
    try:
        return await job_svc.create_project(db, created_by=_uid(user), **payload.model_dump())
    except job_svc.JobError as exc:
        raise HTTPException(400, str(exc)) from None


@router.get("/labelops/projects")
async def list_projects(limit: int = 100, db: AsyncSession = Depends(db_session)):
    return {"projects": await job_svc.list_projects(db, limit)}


@router.get("/labelops/projects/{project_id}/board")
async def project_board(project_id: str, db: AsyncSession = Depends(db_session)):
    """Job counts per stage and state, plus who is currently loaded."""
    return await job_svc.project_board(db, project_id)


@router.post("/labelops/tasks")
async def create_task(payload: TaskIn, _user=Depends(require_role("reviewer")),
                      db: AsyncSession = Depends(db_session)):
    try:
        return await job_svc.create_task(db, **payload.model_dump())
    except job_svc.JobError as exc:
        raise HTTPException(400, str(exc)) from None


# ---- jobs -----------------------------------------------------------------------------------------------

@router.get("/labelops/jobs")
async def list_jobs(project_id: str | None = None, task_id: str | None = None,
                    assignee_id: str | None = None, stage: str | None = None, state: str | None = None,
                    limit: int = 200, db: AsyncSession = Depends(db_session)):
    return {"jobs": await job_svc.list_jobs(db, project_id=project_id, task_id=task_id,
                                            assignee_id=assignee_id, stage=stage, state=state, limit=limit)}


@router.get("/labelops/my-jobs")
async def my_jobs(user=Depends(require_role("annotator")), db: AsyncSession = Depends(db_session)):
    """The caller's own queue: what they should be working on right now."""
    return {"jobs": await job_svc.list_jobs(db, assignee_id=_uid(user))}


@router.post("/labelops/jobs/{job_id}/assign")
async def assign_job(job_id: str, payload: AssignIn, _user=Depends(require_role("reviewer")),
                     db: AsyncSession = Depends(db_session)):
    try:
        return await job_svc.assign_job(db, job_id, payload.assignee_id,
                                        expected_version=payload.expected_version)
    except job_svc.JobError as exc:
        raise HTTPException(409 if "moved on" in str(exc) else 400, str(exc)) from None


@router.post("/labelops/jobs/{job_id}/state")
async def set_state(job_id: str, payload: StateIn, _user=Depends(require_role("annotator")),
                    db: AsyncSession = Depends(db_session)):
    try:
        return await job_svc.set_state(db, job_id, payload.state,
                                       expected_version=payload.expected_version)
    except job_svc.JobError as exc:
        raise HTTPException(409 if "moved on" in str(exc) else 400, str(exc)) from None


@router.post("/labelops/jobs/{job_id}/submit")
async def submit_job(job_id: str, payload: SubmitIn, _user=Depends(require_role("annotator")),
                     db: AsyncSession = Depends(db_session)):
    """Submit the current stage. Honeypots are scored here; failing the project floor sends the job back
    rather than advancing it."""
    try:
        return await job_svc.submit_job(db, job_id, expected_version=payload.expected_version)
    except job_svc.JobError as exc:
        raise HTTPException(409 if "moved on" in str(exc) else 400, str(exc)) from None


# ---- issues ---------------------------------------------------------------------------------------------

@router.post("/labelops/issues")
async def create_issue(payload: IssueIn, user=Depends(require_role("annotator")),
                       db: AsyncSession = Depends(db_session)):
    try:
        return await issue_svc.create_issue(db, created_by=_uid(user), **payload.model_dump())
    except issue_svc.IssueError as exc:
        raise HTTPException(400, str(exc)) from None


@router.get("/labelops/issues")
async def list_issues(frame_id: str | None = None, job_id: str | None = None,
                      object_id: str | None = None, status: str | None = None,
                      db: AsyncSession = Depends(db_session)):
    return {"issues": await issue_svc.list_issues(db, frame_id=frame_id, job_id=job_id,
                                                  object_id=object_id, status=status)}


@router.get("/labelops/issues/{issue_id}")
async def get_issue(issue_id: str, db: AsyncSession = Depends(db_session)):
    try:
        return await issue_svc.get_issue(db, issue_id)
    except issue_svc.IssueError as exc:
        raise HTTPException(404, str(exc)) from None


@router.post("/labelops/issues/{issue_id}/comment")
async def comment(issue_id: str, payload: CommentIn, user=Depends(require_role("annotator")),
                  db: AsyncSession = Depends(db_session)):
    try:
        return await issue_svc.comment(db, issue_id, payload.body, author_id=_uid(user))
    except issue_svc.IssueError as exc:
        raise HTTPException(400, str(exc)) from None


@router.post("/labelops/issues/{issue_id}/resolve")
async def resolve_issue(issue_id: str, reopen: bool = False, user=Depends(require_role("annotator")),
                        db: AsyncSession = Depends(db_session)):
    try:
        return await issue_svc.resolve_issue(db, issue_id, resolved_by=_uid(user), reopen=reopen)
    except issue_svc.IssueError as exc:
        raise HTTPException(400, str(exc)) from None


# ---- scorecards -----------------------------------------------------------------------------------------

@router.get("/labelops/scorecards")
async def scorecards(project_id: str | None = None, db: AsyncSession = Depends(db_session)):
    """Per-annotator throughput and honeypot quality."""
    return {"scorecards": await quality_svc.annotator_scorecards(db, project_id=project_id)}
