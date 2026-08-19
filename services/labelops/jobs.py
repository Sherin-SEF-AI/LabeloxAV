"""Projects, tasks, and assignable jobs: the work-management layer.

A job is a bounded set of frames one person owns. It moves through two orthogonal axes:

    stage:  annotation -> validation -> acceptance    (where in the pipeline)
    state:  new -> in_progress -> completed | rejected (how far within that stage)

Keeping them separate is what lets the board say "in validation, not yet started", which a single collapsed
enum cannot express. Submitting a job advances the stage and resets the state, so the same job row carries its
whole history rather than being cloned per stage.

Frames are referenced by id, never copied: a job is a view over the corpus, so re-labelling a frame elsewhere
is immediately visible in every job that contains it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, LabelJob, LabelProject, LabelTask, User

log = get_logger("labelops_jobs")

# More than a handful of independent passes over the same frames stops being a measurement and starts being
# the budget. Two is the normal choice; three breaks ties without anyone deciding they are the tiebreaker.
MAX_REPLICAS = 5

STAGES = ("annotation", "validation", "acceptance")
STATES = ("new", "in_progress", "completed", "rejected")

# Submitting from a stage moves the work to the next one; acceptance is terminal.
_NEXT_STAGE = {"annotation": "validation", "validation": "acceptance", "acceptance": "acceptance"}


class JobError(RuntimeError):
    """An invalid work-management transition (bad stage/state, stale version, unknown assignee)."""


async def create_project(db: AsyncSession, *, name: str, description: str | None = None,
                         modality: str = "image", honeypot_frac: float = 0.0,
                         min_honeypot_accuracy: float = 0.9, gold_id: str | None = None,
                         created_by: str | None = None) -> dict:
    if not name.strip():
        raise JobError("project name required")
    if not 0.0 <= honeypot_frac <= 0.5:
        raise JobError("honeypot_frac must be between 0 and 0.5")
    p = LabelProject(name=name.strip(), description=description, modality=modality,
                     honeypot_frac=honeypot_frac, min_honeypot_accuracy=min_honeypot_accuracy,
                     gold_id=gold_id, created_by=UUID(created_by) if created_by else None)
    db.add(p)
    try:
        await db.commit()
    except IntegrityError as exc:
        # `label_project.name` is unique. Reusing a name is an ordinary mistake, and letting the constraint
        # surface as a 500 tells the person who made it that the server broke rather than that the name is
        # taken. The rollback matters too: without it the session is poisoned for everything after.
        await db.rollback()
        raise JobError(f"a project named {name.strip()!r} already exists") from exc
    log.info("labelops.project_created", project=str(p.project_id), name=name)
    return _project_dict(p)


def _project_dict(p: LabelProject) -> dict:
    return {"project_id": str(p.project_id), "name": p.name, "description": p.description,
            "modality": p.modality, "honeypot_frac": p.honeypot_frac,
            "min_honeypot_accuracy": p.min_honeypot_accuracy, "gold_id": p.gold_id,
            "label_config": p.label_config or {},
            "created_at": p.created_at.isoformat() if p.created_at else None}


def _job_dict(j: LabelJob) -> dict:
    return {"job_id": str(j.job_id), "task_id": str(j.task_id),
            "assignee_id": str(j.assignee_id) if j.assignee_id else None,
            "stage": j.stage, "state": j.state, "version": j.version,
            "n_frames": len(j.frame_ids or []), "frame_ids": [str(f) for f in (j.frame_ids or [])],
            "replica_group": str(j.replica_group) if j.replica_group else None,
            "replica_index": j.replica_index,
            "n_honeypots": len(j.honeypot_frame_ids or []),
            "honeypot_accuracy": j.honeypot_accuracy, "honeypot_detail": j.honeypot_detail or {},
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "submitted_at": j.submitted_at.isoformat() if j.submitted_at else None,
            "created_at": j.created_at.isoformat() if j.created_at else None}


async def create_task(db: AsyncSession, *, project_id: str, name: str,
                      session_id: str | None = None, predicate: dict | None = None,
                      jobs_of: int = 50, replicas: int = 1) -> dict:
    """Create a task over a session or an explorer predicate, and split its frames into jobs of `jobs_of`.

    The split happens once, at creation, so a job's contents are stable while someone works on it: a frame
    ingested later must not silently appear inside a job already in review.

    `replicas` sends each chunk to that many annotators independently, which is what buys a measurement of
    how much they agree. It costs proportionally: two replicas is twice the annotation spend and twice the
    honeypot budget, so it belongs on a sample of the work rather than all of it. Replica jobs are blind,
    which is what makes the measurement mean anything.
    """
    project = await db.get(LabelProject, UUID(project_id))
    if project is None:
        raise JobError("project not found")
    if jobs_of < 1:
        raise JobError("jobs_of must be at least 1")
    if not 1 <= replicas <= MAX_REPLICAS:
        raise JobError(f"replicas must be between 1 and {MAX_REPLICAS}")

    pred = dict(predicate or {})
    if session_id:
        pred["session_id"] = session_id

    from services.explore.query import frame_select

    stmt = frame_select(pred, Frame.frame_id).order_by(Frame.ts_ns) if pred else select(
        Frame.frame_id).order_by(Frame.ts_ns).limit(1000)
    frame_ids = [str(f) for f in (await db.execute(stmt)).scalars().all()]
    if not frame_ids:
        raise JobError("no frames match this task definition")

    task = LabelTask(project_id=project.project_id, name=name,
                     session_id=UUID(session_id) if session_id else None, predicate=pred,
                     replicas=replicas)
    db.add(task)
    await db.flush()

    jobs = []
    for i in range(0, len(frame_ids), jobs_of):
        chunk = frame_ids[i:i + jobs_of]
        # One group id per chunk, so two jobs over the same frames know they are a pair. The pairing cannot
        # be recovered later by comparing frame_ids: seed_honeypots appends gold frames chosen per job id,
        # so the two lists diverge before this function returns.
        group = uuid.uuid4() if replicas > 1 else None
        for r in range(replicas):
            job = LabelJob(task_id=task.task_id, frame_ids=list(chunk), stage="annotation", state="new",
                           replica_group=group, replica_index=r)
            db.add(job)
            jobs.append(job)
    await db.flush()

    # Seed hidden gold frames per job when the project asks for it.
    seeded = 0
    if project.honeypot_frac > 0 and project.gold_id:
        from services.labelops.quality import seed_honeypots

        for job in jobs:
            seeded += await seed_honeypots(db, job, project)

    await db.commit()
    log.info("labelops.task_created", task=str(task.task_id), frames=len(frame_ids),
             jobs=len(jobs), replicas=replicas, honeypots=seeded)
    return {"task_id": str(task.task_id), "project_id": project_id, "name": name,
            # Distinct frames, not frames times replicas. A board reporting 2x the work because the same
            # frames went to two people would misprice every task.
            "n_frames": len(frame_ids), "n_jobs": len(jobs), "replicas": replicas,
            "honeypots_seeded": seeded, "jobs": [_job_dict(j) for j in jobs]}


async def assign_job(db: AsyncSession, job_id: str, assignee_id: str | None,
                     *, expected_version: int | None = None) -> dict:
    """Assign or unassign a job. Optimistic: a stale expected_version is rejected rather than overwriting
    someone else's assignment."""
    job = await db.get(LabelJob, UUID(job_id))
    if job is None:
        raise JobError("job not found")
    if expected_version is not None and job.version != expected_version:
        raise JobError(f"job moved on (version {job.version}, expected {expected_version})")
    if assignee_id:
        user = await db.get(User, UUID(assignee_id))
        if user is None:
            raise JobError("assignee not found")
        if job.replica_group is not None:
            # One person labelling both replicas is one opinion recorded twice, and it would read as
            # perfect agreement.
            sibling = (await db.execute(
                select(LabelJob).where(LabelJob.replica_group == job.replica_group,
                                       LabelJob.job_id != job.job_id,
                                       LabelJob.assignee_id == user.user_id))).scalars().first()
            if sibling is not None:
                raise JobError(
                    "this annotator already holds the other replica of these frames; agreement between "
                    "one person and themselves is not a measurement")
        job.assignee_id = user.user_id
    else:
        job.assignee_id = None
    job.version += 1
    await db.commit()
    log.info("labelops.job_assigned", job=job_id, assignee=assignee_id)
    from services.integrations.webhooks import emit

    await emit("job.assigned", {"job_id": job_id, "assignee_id": assignee_id})

    # Addressed to the person, not the role: this one has an owner, and telling everyone would be noise for
    # everyone except them.
    if assignee_id:
        from services.notify import notify

        await notify(db, kind="job_assigned", user_id=assignee_id,
                     title=f"a labeling job was assigned to you ({job.stage})",
                     body=f"{len(job.frame_ids or [])} frames to label",
                     href="/projects", subject_type="label_job", subject_id=job_id)
    return _job_dict(job)


async def set_state(db: AsyncSession, job_id: str, state: str,
                    *, expected_version: int | None = None) -> dict:
    """Move a job within its current stage."""
    if state not in STATES:
        raise JobError(f"state must be one of {STATES}")
    job = await db.get(LabelJob, UUID(job_id))
    if job is None:
        raise JobError("job not found")
    if expected_version is not None and job.version != expected_version:
        raise JobError(f"job moved on (version {job.version}, expected {expected_version})")
    if state == "in_progress" and job.started_at is None:
        job.started_at = datetime.now(UTC)
    job.state = state
    job.version += 1
    await db.commit()
    return _job_dict(job)


async def submit_job(db: AsyncSession, job_id: str, *, expected_version: int | None = None) -> dict:
    """Submit the current stage: score any honeypots, then advance the stage and reset the state.

    A job whose honeypot accuracy is below the project floor is sent back as `rejected` in the SAME stage
    rather than advanced. Letting it through and flagging it later would put work that failed its own quality
    bar into the reviewer's queue as though it had passed.
    """
    job = await db.get(LabelJob, UUID(job_id))
    if job is None:
        raise JobError("job not found")
    if expected_version is not None and job.version != expected_version:
        raise JobError(f"job moved on (version {job.version}, expected {expected_version})")

    task = await db.get(LabelTask, job.task_id)
    project = await db.get(LabelProject, task.project_id) if task else None

    result: dict = {}
    if job.honeypot_frame_ids:
        from services.labelops.quality import score_honeypots

        result = await score_honeypots(db, job, project)
        job.honeypot_accuracy = result.get("accuracy")
        job.honeypot_detail = result

    floor = project.min_honeypot_accuracy if project else 0.0
    failed = (job.honeypot_accuracy is not None and job.honeypot_accuracy < floor)

    job.submitted_at = datetime.now(UTC)
    if failed:
        job.state = "rejected"          # stays in the same stage: the work comes back to its author
    else:
        job.stage = _NEXT_STAGE.get(job.stage, "acceptance")
        job.state = "completed" if job.stage == "acceptance" else "new"
    job.version += 1
    await db.commit()

    # A submitted blind-audit job means the auditor is done looking at those frames. Until that is on
    # record the audit cannot be scored, because a frame with no boxes is otherwise indistinguishable
    # from a frame nobody opened, and scoring the second as "the human found nothing" would report the
    # model's recall as perfect precisely where it was being tested. Only on the way forward: a rejected
    # job comes back to its author and is not finished with.
    if not failed:
        from services.verdyx.blind_audit import audit_for_job, mark_frames_labeled

        audit = await audit_for_job(db, job.job_id)
        if audit is not None:
            n = await mark_frames_labeled(db, audit.audit_id,
                                          [UUID(str(f)) for f in (job.frame_ids or [])])
            log.info("labelops.audit_frames_marked", job=job_id, audit=str(audit.audit_id), frames=n)
    log.info("labelops.job_submitted", job=job_id, stage=job.stage, state=job.state,
             honeypot_accuracy=job.honeypot_accuracy, failed=failed)
    # Let external pipelines react without polling. Fire and forget: a slow receiver must not hold up the
    # annotator who just clicked submit.
    from services.integrations.webhooks import emit

    await emit("job.rejected" if failed else "job.submitted",
               {"job_id": job_id, "stage": job.stage, "state": job.state,
                "honeypot_accuracy": job.honeypot_accuracy},
               project_id=str(task.project_id) if task else None)
    return {**_job_dict(job), "honeypot_failed": failed, "min_honeypot_accuracy": floor}


async def list_projects(db: AsyncSession, limit: int = 100) -> list[dict]:
    rows = (await db.execute(
        select(LabelProject).order_by(LabelProject.created_at.desc()).limit(limit))).scalars().all()
    return [_project_dict(p) for p in rows]


async def list_jobs(db: AsyncSession, *, project_id: str | None = None, task_id: str | None = None,
                    assignee_id: str | None = None, stage: str | None = None, state: str | None = None,
                    limit: int = 200) -> list[dict]:
    stmt = select(LabelJob)
    if task_id:
        stmt = stmt.where(LabelJob.task_id == UUID(task_id))
    if project_id:
        stmt = stmt.join(LabelTask, LabelTask.task_id == LabelJob.task_id).where(
            LabelTask.project_id == UUID(project_id))
    if assignee_id:
        stmt = stmt.where(LabelJob.assignee_id == UUID(assignee_id))
    if stage:
        stmt = stmt.where(LabelJob.stage == stage)
    if state:
        stmt = stmt.where(LabelJob.state == state)
    rows = (await db.execute(stmt.order_by(LabelJob.created_at).limit(limit))).scalars().all()
    return [_job_dict(j) for j in rows]


async def project_board(db: AsyncSession, project_id: str) -> dict:
    """Job counts per (stage, state) for the project board, plus per-assignee load."""
    rows = (await db.execute(
        select(LabelJob.stage, LabelJob.state, func.count())
        .join(LabelTask, LabelTask.task_id == LabelJob.task_id)
        .where(LabelTask.project_id == UUID(project_id))
        .group_by(LabelJob.stage, LabelJob.state))).all()
    load = (await db.execute(
        select(User.name, func.count())
        .join(LabelJob, LabelJob.assignee_id == User.user_id)
        .join(LabelTask, LabelTask.task_id == LabelJob.task_id)
        .where(LabelTask.project_id == UUID(project_id), LabelJob.state.in_(("new", "in_progress")))
        .group_by(User.name).order_by(func.count().desc()))).all()
    return {"project_id": project_id,
            "cells": [{"stage": s, "state": st, "count": int(n)} for s, st, n in rows],
            "open_load": [{"assignee": n, "open_jobs": int(c)} for n, c in load]}
