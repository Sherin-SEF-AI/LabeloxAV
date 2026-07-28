"""Campaigns: the improvement loop, run by the system instead of by somebody remembering to.

Every stage of improving a stuck class has existed for a while, and a person stood between each pair of
them. Read the gate's per-class deficit. Mine the objects worth reviewing. Run the VLM judge over them.
Wait for the humans. Launch a retrain. Attempt promotion. Read the result. Decide whether to go again.
That is precisely the work the flywheel was built to remove, and it stayed manual, so a class stalled the
moment nobody was watching. Iteration 6 cleared cattle from 0.14 to 0.59 through exactly this sequence,
driven by hand.

A campaign is the orchestration. Three constraints shape it, and each exists because the failure it
prevents is worse than the work it saves:

- **A budget in labels, not in time.** The batches this builds are human hours. An autonomous loop that
  can commission unbounded review is a way to spend a team, and no wall-clock limit constrains that.
- **A stopping condition that is not "the target".** A campaign that can only stop by succeeding cannot
  stop. Patience counts consecutive iterations that did not move the metric, so a class that is not
  responding is abandoned rather than ground against forever.
- **A human at every gate by default.** `require_approval` starts true. Stages are opted into autopilot
  one at a time, and promotion is the one nobody should opt in casually: a loop that can promote with no
  person in it is a different product with a different risk profile.

The runner is a state machine advanced one step per call rather than a long-lived task. A loop that owns a
process cannot survive a restart, cannot be inspected halfway, and cannot be stopped except by killing
something; a machine that takes one step per tick can do all three.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Campaign, CampaignStep

log = get_logger("campaign")

# The sequence, in order. A campaign at stage N advances to N+1 when that stage completes.
STAGES = ("mine", "judge", "label", "train", "evaluate", "promote")

TERMINAL = {"succeeded", "exhausted", "stopped"}


class CampaignError(Exception):
    """A campaign operation refused, with a message safe to show an operator."""


# ---------------------------------------------------------------- lifecycle

async def create_campaign(db: AsyncSession, *, name: str, class_name: str,
                          target_metric: str = "recall", target_value: float = 0.6,
                          label_budget: int = 2000, max_iterations: int = 6,
                          patience: int = 2, task_type: str = "detection",
                          require_approval: bool = True,
                          autopilot_stages: list[str] | None = None,
                          created_by: str | None = None, notes: str | None = None) -> dict:
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    if not onto.has_name(class_name):
        raise CampaignError(f"{class_name!r} is not in the ontology")
    if label_budget <= 0:
        raise CampaignError("a campaign needs a label budget; an unbounded one can commission a team")
    if not 0 < target_value <= 1:
        raise CampaignError("target_value must be a fraction between 0 and 1")

    bad = [s for s in (autopilot_stages or []) if s not in STAGES]
    if bad:
        raise CampaignError(f"unknown autopilot stage(s) {bad}; known: {list(STAGES)}")

    existing = (await db.execute(
        select(Campaign).where(Campaign.name == name))).scalar_one_or_none()
    if existing is not None:
        raise CampaignError(f"a campaign named {name!r} already exists")

    row = Campaign(name=name, class_name=class_name, task_type=task_type,
                   target_metric=target_metric, target_value=float(target_value),
                   label_budget=int(label_budget), max_iterations=int(max_iterations),
                   patience=int(patience), require_approval=require_approval,
                   autopilot_stages=list(autopilot_stages or []),
                   created_by=created_by, notes=notes, status="pending")
    db.add(row)
    await db.commit()
    await db.refresh(row)
    log.info("campaign.created", name=name, cls=class_name, budget=label_budget)
    return _dict(row)


async def stop_campaign(db: AsyncSession, campaign_id: str, reason: str = "stopped by an operator") -> dict:
    c = await _get(db, campaign_id)
    if c.status in TERMINAL:
        return {**_dict(c), "detail": f"already {c.status}"}
    c.status = "stopped"
    c.updated_at = datetime.now(UTC)
    db.add(CampaignStep(campaign_id=c.campaign_id, iteration=c.iteration, stage="promote",
                        status="stopped", detail={"reason": reason},
                        finished_at=datetime.now(UTC)))
    await db.commit()
    log.info("campaign.stopped", name=c.name, reason=reason)
    return _dict(c)


# ---------------------------------------------------------------- the state machine

async def tick(db: AsyncSession, campaign_id: str, *, dry_run: bool = False) -> dict:
    """Advance a campaign by one step, and report what it did or what it is waiting for.

    One step per call, deliberately. A long-lived task cannot survive a restart, cannot be inspected
    halfway, and cannot be stopped except by killing something.
    """
    c = await _get(db, campaign_id)
    if c.status in TERMINAL:
        return {"campaign": _dict(c), "action": "none", "detail": f"campaign is {c.status}"}

    stop = _stop_reason(c)
    if stop:
        c.status = stop[0]
        await db.commit()
        return {"campaign": _dict(c), "action": "halted", "detail": stop[1]}

    stage = await _next_stage(db, c)
    waiting = await _blocking_step(db, c)
    if waiting is not None:
        return {"campaign": _dict(c), "action": "waiting", "stage": waiting.stage,
                "awaiting": waiting.awaiting,
                "detail": f"{waiting.stage} is waiting on {waiting.awaiting}"}

    if c.require_approval and stage not in set(c.autopilot_stages or []):
        # Recorded as a waiting step rather than silently doing nothing, so the campaign board shows a
        # queue of decisions rather than a row that has apparently stopped for no reason.
        step = await _open_step(db, c, stage, awaiting=f"approval to run {stage}")
        return {"campaign": _dict(c), "action": "awaiting_approval", "stage": stage,
                "step_id": str(step.step_id),
                "detail": f"{stage} needs approval; approve it or add it to autopilot_stages"}

    if dry_run:
        return {"campaign": _dict(c), "action": "would_run", "stage": stage}

    return await run_stage(db, campaign_id, stage)


async def run_stage(db: AsyncSession, campaign_id: str, stage: str) -> dict:
    """Execute one stage. Called by tick, or directly when an operator approves a waiting step."""
    if stage not in STAGES:
        raise CampaignError(f"unknown stage {stage!r}")
    c = await _get(db, campaign_id)
    if c.status in TERMINAL:
        raise CampaignError(f"campaign is {c.status}")

    if c.status == "pending":
        c.status = "running"
        c.iteration = max(1, c.iteration)

    step = await _open_step(db, c, stage)
    try:
        handler = {
            "mine": _stage_mine, "judge": _stage_judge, "label": _stage_label,
            "train": _stage_train, "evaluate": _stage_evaluate, "promote": _stage_promote,
        }[stage]
        result = await handler(db, c, step)
    except Exception as exc:  # noqa: BLE001
        step.status = "failed"
        step.detail = {**(step.detail or {}), "error": f"{type(exc).__name__}: {exc}"}
        step.finished_at = datetime.now(UTC)
        c.status = "blocked"
        await db.commit()
        log.warning("campaign.stage_failed", name=c.name, stage=stage, error=str(exc))
        return {"campaign": _dict(c), "action": "failed", "stage": stage, "detail": str(exc)}

    step.status = result.get("status", "done")
    step.detail = {**(step.detail or {}), **result.get("detail", {})}
    step.metrics = result.get("metrics", {})
    step.awaiting = result.get("awaiting")
    if step.status != "waiting":
        step.finished_at = datetime.now(UTC)
    c.updated_at = datetime.now(UTC)
    await db.commit()
    log.info("campaign.stage", name=c.name, stage=stage, status=step.status)
    return {"campaign": _dict(c), "action": "ran", "stage": stage,
            "step": _step_dict(step), **{k: v for k, v in result.items() if k == "detail"}}


# ---------------------------------------------------------------- the stages

async def _stage_mine(db: AsyncSession, c: Campaign, step: CampaignStep) -> dict:
    """Rank the objects of the target class worth reviewing, within what the budget still allows."""
    from services.flywheel.gate_directed import mine_for_class

    remaining = max(0, int(c.label_budget) - int(c.labels_spent))
    if remaining <= 0:
        return {"status": "done", "detail": {"exhausted": True, "remaining": 0}}

    # Per-iteration slice of the budget rather than all of it: spending the whole allowance on the first
    # batch removes the campaign's ability to react to what that batch teaches.
    slice_size = max(25, min(remaining, int(c.label_budget) // max(1, int(c.max_iterations))))
    mined = await mine_for_class(db, c.class_name, slice_size)
    return {
        "status": "done",
        "detail": {"objects": len(mined.get("objects") or []),
                   "frames": len(mined.get("frame_ids") or []),
                   "pool": mined.get("pool"),
                   # Said out loud: an exhausted pool means mining is no longer the bottleneck, which is a
                   # different problem from a class that is not learning.
                   "pool_exhausted": bool(mined.get("exhausted")),
                   "object_ids": [o["object_id"] for o in (mined.get("objects") or [])][:2000],
                   "frame_ids": (mined.get("frame_ids") or [])[:2000]},
        "metrics": {"mined": len(mined.get("objects") or [])},
    }


async def _stage_judge(db: AsyncSession, c: Campaign, step: CampaignStep) -> dict:
    """Run the VLM judge over the mined frames, so humans see a pre-sorted batch.

    The judge is not the reviewer. It reorders and flags; every label it touches is still a proposal. That
    distinction is the one iteration 5 got wrong, when training on unreviewed machine labels drove
    pedestrian recall from 0.73 to 0.004.
    """
    mine = await _last_step(db, c, "mine")
    frame_ids = ((mine.detail or {}).get("frame_ids") or []) if mine else []
    if not frame_ids:
        return {"status": "done", "detail": {"skipped": "nothing was mined"}}

    sessions = await _sessions_for_frames(db, frame_ids)
    if not sessions:
        return {"status": "done", "detail": {"skipped": "mined frames resolve to no session"}}

    from services.intelligence.vlm_qa import vlm_qa_session

    judged = 0
    errors: list[str] = []
    # Bounded: a campaign that queued a VLM pass over forty sessions would occupy the GPU the retrain
    # later needs, and the point of judging is to sort one batch.
    for sid in sessions[:5]:
        try:
            out = await vlm_qa_session(uuid.UUID(sid), 60)
            judged += int((out or {}).get("checked") or 0)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{sid[:8]}: {type(exc).__name__}")
    return {"status": "done",
            "detail": {"sessions": len(sessions[:5]), "judged": judged, "errors": errors},
            "metrics": {"judged": judged}}


async def _stage_label(db: AsyncSession, c: Campaign, step: CampaignStep) -> dict:
    """Materialise the review batch and wait for the humans.

    The only stage that always waits, whatever the autopilot settings say. There is no version of this
    system in which a machine supplies the human review, and pretending otherwise is exactly the failure
    the gate exists to catch.
    """
    from services.flywheel.gate_directed import mine_for_class
    from services.notify import notify

    mine = await _last_step(db, c, "mine")
    object_ids = ((mine.detail or {}).get("object_ids") or []) if mine else []
    if not object_ids:
        return {"status": "done", "detail": {"skipped": "nothing to label"}}

    reviewed, pending = await _review_progress(db, object_ids)
    if pending == 0:
        c.labels_spent = int(c.labels_spent) + reviewed
        return {"status": "done", "detail": {"reviewed": reviewed},
                "metrics": {"labels_spent": int(c.labels_spent)}}

    if step.awaiting is None:
        await notify(db, kind="gate_batch_ready", severity="info",
                     title=f"campaign {c.name}: {pending} objects to review",
                     body=f"class {c.class_name}, iteration {c.iteration}",
                     href=f"/review/rapid?campaign={c.campaign_id}",
                     subject_type="campaign", subject_id=str(c.campaign_id))
    _ = mine_for_class  # the batch was mined in the previous stage; named here for the reader
    return {"status": "waiting", "awaiting": f"{pending} objects to be reviewed",
            "detail": {"reviewed": reviewed, "pending": pending},
            "metrics": {"reviewed": reviewed, "pending": pending}}


async def _stage_train(db: AsyncSession, c: Campaign, step: CampaignStep) -> dict:
    """Launch a retrain, or report on the one already running."""
    from db.models import TrainingJob

    if step.job_id:
        job = await db.get(TrainingJob, step.job_id)
        if job and job.status in ("done", "failed", "cancelled"):
            return {"status": "done" if job.status == "done" else "failed",
                    "detail": {"job_id": str(job.job_id), "job_status": job.status},
                    "metrics": (job.metrics or {}).get("candidate") or {}}
        return {"status": "waiting", "awaiting": "the retrain to finish",
                "detail": {"job_id": str(step.job_id)}}

    running = (await db.execute(
        select(TrainingJob).where(TrainingJob.status == "running").limit(1))).scalars().first()
    if running is not None:
        # One GPU, held by a Postgres advisory lock. Queuing a second job would not make it run sooner.
        return {"status": "waiting", "awaiting": "the GPU, held by another training job",
                "detail": {"blocked_by": str(running.job_id)}}

    from services.training.jobs import TrainJobSpec, enqueue_job

    # Trained on reviewed labels only. Iteration 5 trained on unreviewed machine labels and drove
    # pedestrian recall from 0.73 to 0.004, which is the whole reason the label stage waits for humans.
    job_id = await enqueue_job(TrainJobSpec(
        purpose=f"campaign-{c.name}", task_type=c.task_type,
        dataset_spec={"name": f"campaign-{c.name}-i{c.iteration}",
                      "states": ["accepted", "approved"]},
        hparams={}, notes=f"campaign {c.name} iteration {c.iteration} for {c.class_name}"))
    step.job_id = uuid.UUID(str(job_id))
    return {"status": "waiting", "awaiting": "the retrain to finish",
            "detail": {"job_id": str(step.job_id)}}


async def _stage_evaluate(db: AsyncSession, c: Campaign, step: CampaignStep) -> dict:
    """Read the target metric for the target class off the finished run, and decide whether it moved."""
    train = await _last_step(db, c, "train")
    metrics = dict((train.metrics or {}) if train else {})
    value = _class_metric(metrics, c.class_name, c.target_metric)

    if value is None:
        # Not scored as zero. A run that produced no per-class number has not regressed to nothing; it has
        # failed to report, and treating the two the same would exhaust patience on a measurement problem.
        return {"status": "done",
                "detail": {"unmeasured": True,
                           "note": f"the run reported no {c.target_metric} for {c.class_name}"},
                "metrics": metrics}

    improved = c.best_value is None or value > float(c.best_value) + 1e-6
    if improved:
        c.best_value = float(value)
        c.stalled_iterations = 0
    else:
        c.stalled_iterations = int(c.stalled_iterations) + 1

    if value >= float(c.target_value):
        c.status = "succeeded"
    return {"status": "done",
            "detail": {"value": round(float(value), 4), "improved": improved,
                       "target": float(c.target_value),
                       "stalled_iterations": int(c.stalled_iterations)},
            "metrics": {c.target_metric: float(value), **metrics}}


async def _stage_promote(db: AsyncSession, c: Campaign, step: CampaignStep) -> dict:
    """Attempt promotion, then start the next iteration or stop.

    The promotion itself goes through the ordinary gate. A campaign gets no special path: the whole point
    of the gate is that it is the same for a human and for a loop.
    """
    train = await _last_step(db, c, "train")
    job_id = (train.detail or {}).get("job_id") if train else None

    promoted = False
    detail: dict = {"job_id": job_id}
    if job_id:
        try:
            from services.govern.champion import evaluate_and_promote

            model_version = (train.metrics or {}).get("model_version") or f"job:{job_id}"
            result = await evaluate_and_promote(db, str(model_version), c.task_type)
            promoted = bool(result.get("promoted"))
            detail.update({"promoted": promoted, "reason": result.get("reason")})
        except Exception as exc:  # noqa: BLE001
            # A refused promotion is an outcome, not a campaign failure: it is the gate doing its job, and
            # the campaign's response is another iteration rather than a stop.
            detail.update({"promoted": False, "gate_error": f"{type(exc).__name__}: {exc}"})

    if c.status != "succeeded":
        c.iteration = int(c.iteration) + 1
    return {"status": "done", "detail": detail, "metrics": {"promoted": 1.0 if promoted else 0.0}}


# ---------------------------------------------------------------- helpers

def _stop_reason(c: Campaign) -> tuple[str, str] | None:
    if c.best_value is not None and float(c.best_value) >= float(c.target_value):
        return ("succeeded", f"{c.target_metric} reached {c.best_value:.3f}")
    if int(c.iteration) > int(c.max_iterations):
        return ("exhausted", f"ran {c.max_iterations} iterations without reaching the target")
    if int(c.labels_spent) >= int(c.label_budget):
        return ("exhausted", f"spent the label budget of {c.label_budget}")
    if int(c.stalled_iterations) >= int(c.patience):
        return ("exhausted",
                f"{c.stalled_iterations} iterations without improvement; the class is not responding")
    return None


async def _next_stage(db: AsyncSession, c: Campaign) -> str:
    """The stage this iteration has not yet completed."""
    done = {s.stage for s in (await db.execute(
        select(CampaignStep).where(CampaignStep.campaign_id == c.campaign_id,
                                   CampaignStep.iteration == max(1, c.iteration),
                                   CampaignStep.status == "done"))).scalars().all()}
    for stage in STAGES:
        if stage not in done:
            return stage
    return STAGES[0]


async def _blocking_step(db: AsyncSession, c: Campaign) -> CampaignStep | None:
    return (await db.execute(
        select(CampaignStep).where(CampaignStep.campaign_id == c.campaign_id,
                                   CampaignStep.iteration == max(1, c.iteration),
                                   CampaignStep.status == "waiting")
        .order_by(CampaignStep.started_at.desc()).limit(1))).scalars().first()


async def _open_step(db: AsyncSession, c: Campaign, stage: str,
                     awaiting: str | None = None) -> CampaignStep:
    existing = (await db.execute(
        select(CampaignStep).where(CampaignStep.campaign_id == c.campaign_id,
                                   CampaignStep.iteration == max(1, c.iteration),
                                   CampaignStep.stage == stage,
                                   CampaignStep.status.in_(("running", "waiting")))
        .order_by(CampaignStep.started_at.desc()).limit(1))).scalars().first()
    if existing is not None:
        if awaiting:
            existing.awaiting = awaiting
            existing.status = "waiting"
            await db.commit()
        return existing
    step = CampaignStep(campaign_id=c.campaign_id, iteration=max(1, c.iteration), stage=stage,
                        status="waiting" if awaiting else "running", awaiting=awaiting)
    db.add(step)
    await db.commit()
    await db.refresh(step)
    return step


async def _last_step(db: AsyncSession, c: Campaign, stage: str) -> CampaignStep | None:
    return (await db.execute(
        select(CampaignStep).where(CampaignStep.campaign_id == c.campaign_id,
                                   CampaignStep.iteration == max(1, c.iteration),
                                   CampaignStep.stage == stage)
        .order_by(CampaignStep.started_at.desc()).limit(1))).scalars().first()


async def _sessions_for_frames(db: AsyncSession, frame_ids: list[str]) -> list[str]:
    from db.models import Frame

    ids = [uuid.UUID(f) for f in frame_ids[:2000]]
    if not ids:
        return []
    rows = (await db.execute(
        select(Frame.session_id).where(Frame.frame_id.in_(ids)).distinct())).scalars().all()
    return [str(s) for s in rows]


async def _review_progress(db: AsyncSession, object_ids: list[str]) -> tuple[int, int]:
    """How much of the batch a human has actually decided."""
    from db.models import Object

    ids = [uuid.UUID(o) for o in object_ids[:5000]]
    if not ids:
        return 0, 0
    total = len(ids)
    reviewed = (await db.execute(
        select(func.count()).select_from(Object)
        .where(Object.object_id.in_(ids),
               Object.state.in_(("accepted", "approved", "rejected"))))).scalar_one()
    return int(reviewed), max(0, total - int(reviewed))


def _class_metric(metrics: dict, class_name: str, metric: str) -> float | None:
    """Pull one class's metric out of a run's per-class block, tolerating either shape it comes in."""
    per_class = metrics.get("per_class_pr") or metrics.get("per_class") or {}
    entry = per_class.get(class_name)
    if isinstance(entry, dict):
        value = entry.get(metric)
        return float(value) if isinstance(value, int | float) else None
    if isinstance(entry, int | float):
        return float(entry)
    return None


async def _get(db: AsyncSession, campaign_id: str) -> Campaign:
    try:
        row = await db.get(Campaign, uuid.UUID(str(campaign_id)))
    except (ValueError, AttributeError):
        row = (await db.execute(
            select(Campaign).where(Campaign.name == str(campaign_id)))).scalar_one_or_none()
    if row is None:
        raise CampaignError(f"campaign {campaign_id!r} not found")
    return row


async def list_campaigns(db: AsyncSession, *, status: str | None = None, limit: int = 100) -> dict:
    stmt = select(Campaign).order_by(Campaign.created_at.desc()).limit(min(max(limit, 1), 500))
    if status:
        stmt = stmt.where(Campaign.status == status)
    rows = (await db.execute(stmt)).scalars().all()
    return {"campaigns": [_dict(r) for r in rows]}


async def campaign_detail(db: AsyncSession, campaign_id: str) -> dict:
    c = await _get(db, campaign_id)
    steps = (await db.execute(
        select(CampaignStep).where(CampaignStep.campaign_id == c.campaign_id)
        .order_by(CampaignStep.iteration, CampaignStep.started_at))).scalars().all()
    return {**_dict(c), "steps": [_step_dict(s) for s in steps],
            "next_stage": await _next_stage(db, c),
            "halt_reason": (_stop_reason(c) or (None, None))[1]}


def _dict(c: Campaign) -> dict:
    return {
        "campaign_id": str(c.campaign_id), "name": c.name, "class_name": c.class_name,
        "task_type": c.task_type, "target_metric": c.target_metric,
        "target_value": float(c.target_value), "label_budget": int(c.label_budget),
        "labels_spent": int(c.labels_spent), "max_iterations": int(c.max_iterations),
        "patience": int(c.patience), "status": c.status, "iteration": int(c.iteration),
        "stalled_iterations": int(c.stalled_iterations),
        "best_value": float(c.best_value) if c.best_value is not None else None,
        "require_approval": bool(c.require_approval),
        "autopilot_stages": list(c.autopilot_stages or []),
        "created_by": c.created_by, "notes": c.notes,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
    }


def _step_dict(s: CampaignStep) -> dict:
    return {"step_id": str(s.step_id), "iteration": int(s.iteration), "stage": s.stage,
            "status": s.status, "detail": s.detail or {}, "metrics": s.metrics or {},
            "awaiting": s.awaiting, "job_id": str(s.job_id) if s.job_id else None,
            "started_at": s.started_at.isoformat() if s.started_at else None,
            "finished_at": s.finished_at.isoformat() if s.finished_at else None}
