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
    # More than one sends each chunk of frames to that many annotators independently, which is what buys a
    # measurement of how much they agree. It costs proportionally, so it belongs on a sample.
    replicas: int = 1


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


@router.post("/labelops/replica-groups/{replica_group}/agreement",
             dependencies=[Depends(require_role("reviewer"))])
async def score_agreement(replica_group: str, iou_thresh: float = 0.5,
                          db: AsyncSession = Depends(db_session)):
    """Compare the replicas of one group and open an issue on every object they disagree about."""
    from services.labelops.agreement import AgreementError, score_replica_group

    try:
        return await score_replica_group(db, replica_group, iou_thresh=iou_thresh)
    except AgreementError as exc:
        raise HTTPException(400, str(exc)) from None


@router.get("/labelops/tasks/{task_id}/agreement", dependencies=[Depends(require_role("annotator"))])
async def task_agreement(task_id: str, db: AsyncSession = Depends(db_session)):
    """How much the annotators on this task agreed, and which frames to look at first."""
    from services.labelops.agreement import AgreementError, group_agreement

    try:
        return await group_agreement(db, task_id)
    except AgreementError as exc:
        raise HTTPException(404, str(exc)) from None


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
                      mine: bool = False,
                      db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """List issues. `mine=true` returns the ones raised about the caller's own work.

    That dimension did not exist: every filter here was a way of asking what is wrong with a given frame,
    job or object, and none of them a way for an annotator to find what has been said about their labels.
    """
    about = str(user.user_id) if (mine and user is not None) else None
    return {"issues": await issue_svc.list_issues(db, frame_id=frame_id, job_id=job_id,
                                                  object_id=object_id, status=status,
                                                  about_user=about)}


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

@router.get("/labelops/scorecards", dependencies=[Depends(require_role("reviewer"))])
async def scorecards(project_id: str | None = None, db: AsyncSession = Depends(db_session)):
    """Per-annotator throughput and honeypot quality.

    Reviewer-gated, like its neighbours. It was the one route in this section with no role floor, which
    made every annotator's performance record readable by anybody the API let in, including an annotator
    reading their colleagues'. Reading your own record is a different question and has its own route below.
    """
    return {"scorecards": await quality_svc.annotator_scorecards(db, project_id=project_id)}


@router.get("/labelops/scorecards/full", dependencies=[Depends(require_role("reviewer"))])
async def full_scorecards(confidence: float = 0.95, db: AsyncSession = Depends(db_session)):
    """People and vendors in one table, with the per-class record that decides what to send whom."""
    from services.labelops.scorecard import scorecards as build

    return await build(db, confidence=confidence)


@router.get("/labelops/scorecards/me", dependencies=[Depends(require_role("annotator"))])
async def my_scorecard(user=Depends(current_user), db: AsyncSession = Depends(db_session)):
    """Your own record. Everyone can see how they are doing; that is not a ranking of anybody else."""
    from services.labelops.scorecard import scorecard_for

    if user is None:
        raise HTTPException(401, "sign in to see your own scorecard")
    out = await scorecard_for(db, str(user.user_id))
    if out is None:
        raise HTTPException(404, "no such user")
    return out


@router.post("/labelops/precision-batch", dependencies=[Depends(require_role("reviewer"))])
async def make_precision_batch(target: int = 300, db: AsyncSession = Depends(db_session)):
    """Stamp a random sample of machine labels for review, to measure how many of them are correct.

    Distinct from the active-learning queue on purpose. That queue surfaces the hardest objects, which is
    right for improving a model and wrong for measuring one: judging it reports the accuracy of the worst
    objects in the corpus rather than of the corpus. This samples randomly within class instead.
    """
    from services.labelops.precision_batch import build_precision_batch

    return await build_precision_batch(db, target=target)


@router.get("/labelops/precision/{batch_id}")
async def read_precision(batch_id: str, db: AsyncSession = Depends(db_session)):
    """Precision per class and for the corpus, from whatever of the batch has been judged so far.

    Reports both the sample figure and a corpus figure re-weighted by how common each class actually is,
    because the batch over-samples rare classes deliberately and quoting the first as the second is the
    mistake this exists to prevent. Every rate carries a Wilson interval, so a number from 12 objects cannot
    be mistaken for one from 12,000.
    """
    from services.labelops.precision_batch import corpus_precision

    return await corpus_precision(db, batch_id)


@router.post("/labelops/prereview/{batch_id}", dependencies=[Depends(require_role("reviewer"))])
async def run_prereview(batch_id: str, limit: int | None = None,
                        db: AsyncSession = Depends(db_session)):
    """Have a VLM judge every label in a batch, so a person adjudicates instead of judging from scratch.

    A reviewer action because it spends money and writes verdicts. Idempotent: re-running the same judge
    updates rows rather than doubling them, and a different model version stands beside the old one instead
    of overwriting it, which is what lets two judges be compared on the same crops.
    """
    from services.labelops.vlm_review import prereview_batch

    return await prereview_batch(db, batch_id, limit=limit)


@router.get("/labelops/prereview/{batch_id}")
async def read_prereview(batch_id: str, model_version: str | None = None,
                         agreement_batch_id: str | None = None,
                         db: AsyncSession = Depends(db_session)):
    """Machine-judged precision for a batch, and how much that judgement is worth.

    Returns the judge's raw agreement rate and, separately, that rate corrected for the judge's own measured
    error. The two are never collapsed: the raw figure is a blend of how good the labels are and how good
    the judge is, and quoting it as precision embeds the judge's error in every claim downstream. When too
    few objects carry both a machine verdict and a human ruling, the corrected figure is null and the caveat
    says what the raw number actually represents.
    """
    from services.labelops.vlm_review import judged_precision

    return await judged_precision(db, batch_id, model_version=model_version,
                                  agreement_batch_id=agreement_batch_id)


@router.get("/labelops/judge-agreement")
async def read_judge_agreement(model_version: str | None = None, batch_id: str | None = None,
                               db: AsyncSession = Depends(db_session)):
    """How well the machine judge tracks human reviewers, on the objects where both have ruled.

    The scoreboard for the judge itself. Sensitivity and specificity here are what make a machine-derived
    precision quotable, and `usable` is false when the judge has not been compared against enough human
    verdicts to say anything, which is the honest state for most of this corpus today.
    """
    from services.labelops.vlm_review import judge_agreement

    return await judge_agreement(db, model_version=model_version, batch_id=batch_id)
