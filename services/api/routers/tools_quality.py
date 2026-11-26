"""API for the quality annotation tools, agent B: quality (#11 consensus, #12 explained errors, #13 VLM reasons, #14 rare scenes).

Mounted under /api; every path here starts with /tools/quality/.

Consensus (#11), services/labelops/agreement.py:
- `GET  /tools/quality/consensus/tasks`                       tasks with replicas, and how many groups can merge
- `GET  /tools/quality/consensus/tasks/{task_id}`             the board: groups, per-class agreement heatmap
- `GET  /tools/quality/consensus/tasks/{task_id}/disagreements`  the open disagreement issues on its frames
- `POST /tools/quality/consensus/groups/{replica_group}/preview` a dry-run merge of one group (writes nothing)
- `POST /tools/quality/consensus/merge`                       score and merge a task or one group (job)

Explained errors (#12), services/errordetect/explain.py:
- `GET /tools/quality/errors/{candidate_id}/why`              one candidate, explained
- `GET /tools/quality/frames/{frame_id}/why`                  every candidate on a frame (the editor overlay)

VLM reason per label (#13), services/intelligence/vlm_qa.py:
- `POST /tools/quality/vlm/check`                             check up to 5 labels now
- `POST /tools/quality/vlm/check-batch`                       check many labels (GPU job, not in office hours)
- `GET  /tools/quality/vlm/checks?ids=`                       stored checks for objects

Rare scenes (#14), services/intelligence/rare_scenes.py:
- `POST /tools/quality/rare-scenes/rank`                      rank the corpus (job)
- `GET  /tools/quality/rare-scenes/latest`                    the latest finished ranking
- `POST /tools/quality/rare-scenes/search`                    by text or by example frame, with rarity
- `POST /tools/quality/rare-scenes/task`                      a labelling task from selected frames
"""

from __future__ import annotations

import uuid
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import LabelJob, LabelProject, LabelTask, Object, ToolJob
from services.api.deps import current_user, db_session, require_role
from services.api.project_guard import caller_hidden_projects

log = get_logger("tools_quality")

router = APIRouter()


def _job_dict(job: ToolJob) -> dict:
    return {"job_id": str(job.job_id), "kind": job.kind, "status": job.status,
            "session_id": str(job.session_id) if job.session_id else None}


# ---- #11 consensus -----------------------------------------------------------------------------------------

class MergeParams(BaseModel):
    iou_thresh: float = Field(0.5, ge=0.05, le=0.95)
    quorum: int = Field(2, ge=2, le=12)
    include_model: bool = True
    require_human: bool = True


class MergeIn(MergeParams):
    task_id: UUID | None = None
    replica_group: UUID | None = None


@router.get("/tools/quality/consensus/tasks", dependencies=[Depends(require_role("annotator"))])
async def consensus_tasks(limit: Annotated[int, Query(ge=1, le=500)] = 100,
                          db: AsyncSession = Depends(db_session),
                          hidden: set = Depends(caller_hidden_projects)):
    """Tasks sent to more than one annotator, newest first, with how many of their groups can be merged."""
    from services.labelops.agreement import CONSENSUS_SOURCE, SUBMITTED_STATES

    q = (select(LabelTask, LabelProject.name).join(LabelProject, LabelProject.project_id == LabelTask.project_id)
         .where(LabelTask.replicas > 1).order_by(LabelTask.created_at.desc()).limit(limit))
    if hidden:
        q = q.where(LabelTask.project_id.notin_(list(hidden)))
    rows = (await db.execute(q)).all()
    tids = [t.task_id for t, _ in rows]
    jobs = list((await db.execute(select(LabelJob).where(
        LabelJob.task_id.in_(tids), LabelJob.replica_group.isnot(None)))).scalars().all()) if tids else []
    groups: dict = {}
    for j in jobs:
        groups.setdefault(j.task_id, {}).setdefault(j.replica_group, []).append(j)
    cons: dict[str, int] = {}
    if tids:
        tid_col = Object.provenance["consensus"]["task_id"].astext
        for tid, n in (await db.execute(
                select(tid_col, func.count()).where(Object.source == CONSENSUS_SOURCE,
                                                    tid_col.in_([str(t) for t in tids]))
                .group_by(tid_col))).all():
            cons[str(tid)] = int(n)
    out = []
    for t, pname in rows:
        g = groups.get(t.task_id, {})
        ready = sum(1 for js in g.values() if len(js) >= 2 and all(j.state in SUBMITTED_STATES for j in js))
        out.append({"task_id": str(t.task_id), "name": t.name, "project_id": str(t.project_id),
                    "project": pname, "replicas": t.replicas, "n_groups": len(g), "ready_groups": ready,
                    "consensus_objects": cons.get(str(t.task_id), 0),
                    "created_at": t.created_at.isoformat() if t.created_at else None})
    return {"tasks": out}


@router.get("/tools/quality/consensus/tasks/{task_id}", dependencies=[Depends(require_role("annotator"))])
async def consensus_board(task_id: UUID, iou_thresh: Annotated[float, Query(ge=0.05, le=0.95)] = 0.5,
                          quorum: Annotated[int, Query(ge=2, le=12)] = 2, include_model: bool = True,
                          require_human: bool = True, db: AsyncSession = Depends(db_session),
                          hidden: set = Depends(caller_hidden_projects)):
    """Per replica group: its jobs, whether it can merge, and a per-class agreement heatmap (a dry run)."""
    from services.labelops.agreement import AgreementError
    from services.labelops.agreement import consensus_board as board

    await _visible_task(db, task_id, hidden)
    try:
        return await board(db, str(task_id), iou_thresh=iou_thresh, quorum=quorum,
                           include_model=include_model, require_human=require_human)
    except AgreementError as exc:
        raise HTTPException(400, str(exc)) from None


async def _visible_task(db: AsyncSession, task_id: UUID, hidden: set) -> LabelTask:
    task = await db.get(LabelTask, task_id)
    if task is None or (hidden and task.project_id in hidden):
        raise HTTPException(404, "task not found")
    return task


@router.get("/tools/quality/consensus/tasks/{task_id}/disagreements",
            dependencies=[Depends(require_role("annotator"))])
async def consensus_disagreements(task_id: UUID, status: str = "open",
                                  limit: Annotated[int, Query(ge=1, le=1000)] = 200,
                                  db: AsyncSession = Depends(db_session),
                                  hidden: set = Depends(caller_hidden_projects)):
    """The disagreement issues on this task's frames: the queue that opens each frame."""
    from services.labelops.agreement import task_disagreements

    if status not in ("open", "resolved"):
        raise HTTPException(400, "status must be open or resolved")
    await _visible_task(db, task_id, hidden)
    return {"items": await task_disagreements(db, str(task_id), status=status, limit=limit)}


@router.post("/tools/quality/consensus/groups/{replica_group}/preview",
             dependencies=[Depends(require_role("annotator"))])
async def consensus_preview(replica_group: UUID, body: MergeParams, db: AsyncSession = Depends(db_session)):
    """What a merge of this group would do with these settings. Writes nothing."""
    from services.labelops.agreement import AgreementError, merge_replica_group

    try:
        return await merge_replica_group(db, str(replica_group), dry_run=True, **body.model_dump())
    except AgreementError as exc:
        raise HTTPException(400, str(exc)) from None


@router.post("/tools/quality/consensus/merge", dependencies=[Depends(require_role("reviewer"))])
async def consensus_merge(body: MergeIn, db: AsyncSession = Depends(db_session), user=Depends(current_user),
                          hidden: set = Depends(caller_hidden_projects)):
    """Score and merge a task's submitted replica groups (or one group). Writes consensus objects and Issues."""
    from services.tools.registry import enqueue

    if (body.task_id is None) == (body.replica_group is None):
        raise HTTPException(400, "give exactly one of task_id or replica_group")
    if body.task_id is not None:
        await _visible_task(db, body.task_id, hidden)
    else:
        j = (await db.execute(select(LabelJob).where(LabelJob.replica_group == body.replica_group))).scalars().first()
        if j is None:
            raise HTTPException(404, "replica group not found")
        await _visible_task(db, j.task_id, hidden)
    params = body.model_dump(mode="json")
    job = await enqueue(db, "consensus_merge", params, user_id=user.user_id)
    return _job_dict(job)


# ---- #12 explained errors ----------------------------------------------------------------------------------

@router.get("/tools/quality/errors/{candidate_id}/why", dependencies=[Depends(require_role("annotator"))])
async def error_why(candidate_id: UUID, db: AsyncSession = Depends(db_session)):
    """Why a check flagged this label: a sentence, the boxes it compared on the frame, the crops it used."""
    from services.errordetect.explain import explain_candidate

    out = await explain_candidate(db, str(candidate_id))
    if out is None:
        raise HTTPException(404, "candidate not found")
    return out


@router.get("/tools/quality/frames/{frame_id}/why", dependencies=[Depends(require_role("annotator"))])
async def frame_why(frame_id: UUID, status: str = "pending", db: AsyncSession = Depends(db_session)):
    """Every likely error on this frame, explained, for drawing over the editor canvas."""
    from services.errordetect.explain import frame_explanations

    if status not in ("pending", "confirmed_error", "dismissed", "all"):
        raise HTTPException(400, "status must be pending, confirmed_error, dismissed or all")
    return await frame_explanations(db, str(frame_id), status=None if status == "all" else status)


# ---- #13 VLM reason per label ------------------------------------------------------------------------------

class CheckIn(BaseModel):
    object_ids: list[UUID] = Field(min_length=1, max_length=5)


class CheckBatchIn(BaseModel):
    object_ids: list[UUID] | None = Field(None, max_length=400)
    session_id: UUID | None = None
    limit: int = Field(50, ge=1, le=400)


@router.post("/tools/quality/vlm/check", dependencies=[Depends(require_role("annotator"))])
async def vlm_check(body: CheckIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Ask the vision model about up to five labels now: verdict, a one-sentence reason, an alternative."""
    from services.intelligence.vlm_qa import check_labels

    res = await check_labels(db, list(body.object_ids), user_id=str(user.user_id))
    if res["found"] == 0:
        raise HTTPException(404, "none of those objects exist")
    if res["checked"] == 0 and res["failed"]:
        raise HTTPException(503, "the vision model did not answer: " + "; ".join(res["errors"][:2]))
    return res


@router.post("/tools/quality/vlm/check-batch", dependencies=[Depends(require_role("reviewer"))])
async def vlm_check_batch(body: CheckBatchIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Queue a VLM check over many labels. A GPU job; it waits out office hours (10:00-18:00 IST)."""
    from services.tools.registry import enqueue

    if not body.object_ids and body.session_id is None:
        raise HTTPException(400, "give object_ids or a session_id")
    params = {"object_ids": [str(o) for o in body.object_ids] if body.object_ids else None,
              "session_id": str(body.session_id) if body.session_id else None, "limit": body.limit}
    job = await enqueue(db, "vlm_label_check", params, session_id=body.session_id, user_id=user.user_id)
    return _job_dict(job)


@router.get("/tools/quality/vlm/checks", dependencies=[Depends(require_role("annotator"))])
async def vlm_checks(ids: Annotated[str, Query(min_length=1, max_length=40_000)],
                     db: AsyncSession = Depends(db_session)):
    """The stored check per object (comma-separated ids); objects never checked are absent."""
    from services.intelligence.vlm_qa import latest_checks

    try:
        parsed = [uuid.UUID(s.strip()) for s in ids.split(",") if s.strip()][:500]
    except ValueError:
        raise HTTPException(400, "ids must be object uuids") from None
    return {"checks": await latest_checks(db, parsed)}


# ---- #14 rare scenes ---------------------------------------------------------------------------------------

class RankIn(BaseModel):
    space: str = Field("dino", pattern="^(dino|siglip)$")
    k: int = Field(10, ge=1, le=50)
    top: int = Field(300, ge=10, le=2000)
    per_session: int = Field(4, ge=1, le=100)
    per_cluster: int = Field(6, ge=1, le=100)
    session_id: UUID | None = None
    city: str | None = Field(None, max_length=64)


class SearchIn(BaseModel):
    text: str | None = Field(None, max_length=300)
    frame_id: UUID | None = None
    space: str | None = Field(None, pattern="^(dino|siglip)$")
    k: int = Field(48, ge=1, le=120)
    rarest_first: bool = False
    session_id: UUID | None = None


class TaskIn(BaseModel):
    project_id: UUID
    name: str = Field(min_length=1, max_length=160)
    frame_ids: list[UUID] = Field(min_length=1, max_length=2000)
    jobs_of: int = Field(50, ge=1, le=500)


@router.post("/tools/quality/rare-scenes/rank", dependencies=[Depends(require_role("reviewer"))])
async def rare_rank(body: RankIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Rank the corpus by how little else looks like each frame. A CPU job over every frame vector."""
    from services.tools.registry import enqueue

    job = await enqueue(db, "rare_scene_rank", body.model_dump(mode="json"), session_id=body.session_id,
                        user_id=user.user_id)
    return _job_dict(job)


@router.get("/tools/quality/rare-scenes/latest", dependencies=[Depends(require_role("annotator"))])
async def rare_latest(space: str | None = None, db: AsyncSession = Depends(db_session)):
    """The newest finished ranking, plus whether one is running now."""
    q = select(ToolJob).where(ToolJob.kind == "rare_scene_rank")
    done = (await db.execute(q.where(ToolJob.status == "done").order_by(ToolJob.finished_at.desc()).limit(20)
                             )).scalars().all()
    if space:
        done = [j for j in done if (j.params or {}).get("space", "dino") == space]
    running = (await db.execute(q.where(ToolJob.status.in_(("pending", "running")))
                                .order_by(ToolJob.created_at.desc()).limit(1))).scalars().first()
    latest = done[0] if done else None
    return {"job": None if latest is None else {**_job_dict(latest),
                                                 "finished_at": latest.finished_at.isoformat() if latest.finished_at else None},
            "result": latest.result if latest is not None else None,
            "running": _job_dict(running) if running is not None else None}


@router.post("/tools/quality/rare-scenes/search", dependencies=[Depends(require_role("annotator"))])
async def rare_search(body: SearchIn, db: AsyncSession = Depends(db_session)):
    """Frames matching a text query or an example frame, each with how rare it is and why."""
    from services.intelligence.rare_scenes import search

    if not (body.text and body.text.strip()) and body.frame_id is None:
        raise HTTPException(400, "give a text query or an example frame")
    try:
        return await search(db, text=body.text, frame_id=str(body.frame_id) if body.frame_id else None,
                            space=body.space, k=body.k, rarest_first=body.rarest_first,
                            session_id=str(body.session_id) if body.session_id else None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@router.post("/tools/quality/rare-scenes/task", dependencies=[Depends(require_role("reviewer"))])
async def rare_task(body: TaskIn, db: AsyncSession = Depends(db_session),
                    hidden: set = Depends(caller_hidden_projects)):
    """Create a labelling task over the selected frames (services/labelops/jobs.py create_task)."""
    from services.labelops.jobs import JobError, create_task

    if hidden and body.project_id in hidden:
        raise HTTPException(404, "project not found")
    try:
        return await create_task(db, project_id=str(body.project_id), name=body.name,
                                 predicate={"frame_ids": [str(f) for f in dict.fromkeys(body.frame_ids)]},
                                 jobs_of=body.jobs_of)
    except JobError as exc:
        raise HTTPException(400, str(exc)) from None
