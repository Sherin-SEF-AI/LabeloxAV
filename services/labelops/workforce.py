"""Routing labeling work to teams that label for a living, and judging what comes back.

253 of 570,379 objects carry a human verdict. That single figure starves every quality number in the system:
precision is unmeasurable, the gate has nothing to tune against, the error detectors hold 298,529 candidates
and one verdict. The tooling for review is good and the pipeline that generates work is good. What has never
existed is anybody to send the work to: `assign_job` takes a user id, so work reaches a person only when
another person picks their name off a list.

The pieces this stands on already exist and are deliberately reused rather than reimplemented.

**Signing** is the webhook helper, including its replay window. A returned batch is an outside party writing
annotation data into the corpus, so it has to be attributable to one workforce and unforgeable by anyone
else. A second signing scheme would be a second thing to get wrong.

**The SSRF guard** is the webhook one too. A dispatch endpoint is caller-supplied input that this server then
fetches with its own network position, which is the same hole a webhook URL opens.

**The acceptance bar** is the honeypot machinery, which is already deterministic per job id, scored on
return, and sends failed jobs back. All this adds is that the bar lives on the workforce, because it is a
commercial term negotiated per vendor rather than a global constant.

The one genuinely new judgement is routing. A workforce earns work by being accurate, measured the same way
the error detectors are: on the lower bound of its accepted-batch rate rather than the point estimate, so a
vendor cannot buy its way to the front of the queue with three good batches.
"""

from __future__ import annotations

import json
import secrets
import uuid as _uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import LabelJob, LabelProject, LabelTask, Workforce, WorkforceAssignment

log = get_logger("labelops.workforce")

OPEN_STATES = ("dispatched", "returned")
STATES = ("dispatched", "returned", "accepted", "rejected", "expired")

# Below this many decided batches a workforce's accept rate is too wide to route on, exactly as with
# detector precision. Reported as unproven rather than as poor.
MIN_BATCHES_FOR_RATING = 5

# What an unproven workforce is worth when routing. Not 1.0, which would send the newest vendor the most
# work; not 0.0, which would mean no vendor could ever get a first batch and the rating could never exist.
UNPROVEN_WORKFORCE_WEIGHT = 0.5


class WorkforceError(RuntimeError):
    """A dispatch or return could not be processed. Carries a message meant for an operator."""


async def register_workforce(db: AsyncSession, *, name: str, kind: str = "vendor",
                             endpoint: str | None = None, capabilities: dict | None = None,
                             capacity_jobs_per_day: int = 0, min_honeypot_accuracy: float = 0.9,
                             contact: str | None = None) -> dict:
    """Register a workforce and mint its callback secret.

    The secret is generated here and returned once. Accepting a caller-supplied secret would let whoever
    registers a workforce choose a value they already know, and the secret is the only thing standing
    between an outside party and the ability to write annotations attributed to that workforce.
    """
    if endpoint:
        from services.integrations.webhooks import _is_safe_webhook_url

        ok, why = _is_safe_webhook_url(endpoint)
        if not ok:
            # Same reasoning as the webhook guard: a dispatch endpoint is caller-supplied input this server
            # then fetches with its own network position.
            raise WorkforceError(f"refusing dispatch endpoint: {why}")

    secret = secrets.token_urlsafe(32)
    wf = Workforce(name=name, kind=kind, endpoint=endpoint, secret=secret,
                   capabilities=capabilities or {}, capacity_jobs_per_day=capacity_jobs_per_day,
                   min_honeypot_accuracy=min_honeypot_accuracy, contact=contact)
    db.add(wf)
    await db.commit()
    log.info("workforce.registered", name=name, kind=kind, has_endpoint=bool(endpoint))
    # The secret is echoed once, at creation, and never returned by any read path afterwards.
    return {**_wf_dict(wf), "secret": secret}


def _wf_dict(w: Workforce) -> dict:
    return {"workforce_id": str(w.workforce_id), "name": w.name, "kind": w.kind,
            "endpoint": w.endpoint, "active": w.active, "capabilities": w.capabilities,
            "capacity_jobs_per_day": w.capacity_jobs_per_day,
            "min_honeypot_accuracy": w.min_honeypot_accuracy, "contact": w.contact}


def _asg_dict(a: WorkforceAssignment) -> dict:
    return {"assignment_id": str(a.assignment_id), "job_id": str(a.job_id),
            "workforce_id": str(a.workforce_id), "state": a.state, "external_ref": a.external_ref,
            "honeypot_accuracy": a.honeypot_accuracy, "objects_returned": a.objects_returned,
            "reason": a.reason, "detail": a.detail}


async def dispatch_job(db: AsyncSession, *, job_id: str, workforce_id: str,
                       deliver: bool = True) -> dict:
    """Send one job to one workforce, and record that it went.

    Refuses a job that already has an open dispatch. Two workforces labeling the same frames is not
    redundancy, it is two invoices and a merge conflict, and the partial unique index makes that a database
    invariant rather than a check somebody might forget.

    `deliver=False` records the dispatch without calling out, for a pull-based workforce that polls
    `pending_for_workforce` instead of receiving a POST.
    """
    job = await db.get(LabelJob, _uuid.UUID(job_id))
    if job is None:
        raise WorkforceError("job not found")
    wf = await db.get(Workforce, _uuid.UUID(workforce_id))
    if wf is None:
        raise WorkforceError("workforce not found")
    if not wf.active:
        raise WorkforceError(f"workforce '{wf.name}' is not active")

    open_row = (await db.execute(
        select(WorkforceAssignment).where(WorkforceAssignment.job_id == job.job_id,
                                          WorkforceAssignment.state.in_(OPEN_STATES)))).scalar_one_or_none()
    if open_row is not None:
        raise WorkforceError(f"job already dispatched (assignment {open_row.assignment_id}, "
                             f"state {open_row.state})")

    asg = WorkforceAssignment(job_id=job.job_id, workforce_id=wf.workforce_id, state="dispatched")
    db.add(asg)
    job.state = "in_progress"
    job.version += 1
    await db.commit()
    await db.refresh(asg)

    delivered = None
    if deliver and wf.endpoint:
        delivered = await _deliver(wf, _dispatch_payload(job, asg))
        if not delivered.get("ok"):
            # The dispatch row stays. A vendor whose endpoint was briefly down still has the work assigned,
            # and deleting the row on a failed POST would silently drop the job back into an unassigned pool
            # while the vendor might already have received it.
            log.warning("workforce.delivery_failed", workforce=wf.name, job=job_id,
                        error=delivered.get("error"))

    log.info("workforce.dispatched", workforce=wf.name, job=job_id, frames=len(job.frame_ids or []))
    return {**_asg_dict(asg), "delivery": delivered}


def _dispatch_payload(job: LabelJob, asg: WorkforceAssignment) -> dict:
    """What a workforce is told about a job.

    Frame ids and the assignment id, not the honeypot ids. Sending those would tell the vendor exactly which
    frames are being used to score them, which is the one thing that would make the quality gate worthless.
    """
    return {
        "assignment_id": str(asg.assignment_id),
        "job_id": str(job.job_id),
        "stage": job.stage,
        "frame_ids": [str(f) for f in (job.frame_ids or [])],
        "frame_count": len(job.frame_ids or []),
    }


async def _deliver(wf: Workforce, payload: dict) -> dict:
    """POST a dispatch, signed the same way webhooks are, including the replay window."""
    import httpx

    from services.integrations.webhooks import sign

    body = json.dumps(payload, separators=(",", ":")).encode()
    ts = int(datetime.now(UTC).timestamp())
    headers = {"Content-Type": "application/json",
               "X-Labelox-Signature": sign(wf.secret, body, ts),
               "X-Labelox-Workforce": str(wf.workforce_id)}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(wf.endpoint, content=body, headers=headers)
        return {"ok": resp.status_code < 400, "status": resp.status_code}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200]}


async def pending_for_workforce(db: AsyncSession, workforce_id: str, limit: int = 50) -> list[dict]:
    """Open dispatches for a workforce that polls instead of receiving POSTs.

    Not every vendor can host a callback endpoint, and requiring one would exclude exactly the smaller
    outfits an India-focused operation would want to work with.
    """
    rows = (await db.execute(
        select(WorkforceAssignment, LabelJob)
        .join(LabelJob, LabelJob.job_id == WorkforceAssignment.job_id)
        .where(WorkforceAssignment.workforce_id == _uuid.UUID(workforce_id),
               WorkforceAssignment.state == "dispatched")
        .order_by(WorkforceAssignment.dispatched_at)
        .limit(limit))).all()
    return [_dispatch_payload(job, asg) for asg, job in rows]


async def submit_return(db: AsyncSession, *, assignment_id: str, external_ref: str | None = None,
                        objects_returned: int = 0, detail: dict | None = None) -> dict:
    """A workforce says it has finished a job. Scores the honeypots and accepts or rejects.

    The verdict is the honeypot accuracy against the bar on the workforce's own row. A rejected batch sends
    the job back to `new` and closes the assignment, so it can be dispatched again, to the same workforce or
    a different one; the rejected row stays, because a vendor's rejected batches are most of what its rating
    is made of.
    """
    asg = await db.get(WorkforceAssignment, _uuid.UUID(assignment_id))
    if asg is None:
        raise WorkforceError("assignment not found")
    if asg.state not in OPEN_STATES:
        # Returning twice is a vendor retry, not a second batch. Answering with the settled verdict is
        # friendlier than an error and keeps the vendor's bookkeeping consistent with ours.
        return {**_asg_dict(asg), "note": "already decided; this return was ignored"}

    job = await db.get(LabelJob, asg.job_id)
    wf = await db.get(Workforce, asg.workforce_id)
    if job is None or wf is None:
        raise WorkforceError("job or workforce missing for this assignment")

    asg.state = "returned"
    asg.returned_at = datetime.now(UTC)
    asg.external_ref = external_ref
    asg.objects_returned = int(objects_returned)
    asg.detail = {**(asg.detail or {}), **(detail or {})}

    accuracy = await _score(db, job)
    asg.honeypot_accuracy = accuracy
    asg.decided_at = datetime.now(UTC)

    if accuracy is None:
        # No honeypots in this job, so there is nothing to check it against. Accepted, and the reason says
        # so: a batch that passed no gate should not be recorded as one that passed a gate.
        asg.state = "accepted"
        asg.reason = "accepted without a quality check: this job carried no honeypots"
        job.state = "completed"
    elif accuracy >= wf.min_honeypot_accuracy:
        asg.state = "accepted"
        asg.reason = f"honeypot accuracy {accuracy:.3f} at or above the agreed {wf.min_honeypot_accuracy:.3f}"
        job.state = "completed"
    else:
        asg.state = "rejected"
        asg.reason = f"honeypot accuracy {accuracy:.3f} below the agreed {wf.min_honeypot_accuracy:.3f}"
        job.state = "new"          # back into the pool, dispatchable again
        job.assignee_id = None

    job.version += 1
    await db.commit()
    log.info("workforce.returned", workforce=wf.name, job=str(job.job_id), state=asg.state,
             accuracy=accuracy)
    return _asg_dict(asg)


async def _score(db: AsyncSession, job: LabelJob) -> float | None:
    """Honeypot accuracy for a returned job, or None when the job carried no honeypots."""
    if not (job.honeypot_frame_ids or []):
        return None
    from services.labelops.quality import score_honeypots

    task = await db.get(LabelTask, job.task_id)
    project = await db.get(LabelProject, task.project_id) if task else None
    result = await score_honeypots(db, job, project)
    return result.get("accuracy") if isinstance(result, dict) else None


async def workforce_rating(db: AsyncSession, *, confidence: float = 0.95) -> dict:
    """How often each workforce's batches are accepted, with the interval and the count.

    The routing weight is the lower bound rather than the point estimate, for the same reason detector
    precision is: three accepted batches out of three is 1.0 and means very little, while ninety out of a
    hundred is 0.9 and means a great deal. Ranking on the point estimate would send the most work to the
    least proven vendor.
    """
    from services.labelops.sampling import wilson_interval

    rows = (await db.execute(
        select(Workforce.name, Workforce.workforce_id, WorkforceAssignment.state,
               func.count(WorkforceAssignment.assignment_id))
        .join(WorkforceAssignment, WorkforceAssignment.workforce_id == Workforce.workforce_id, isouter=True)
        .group_by(Workforce.name, Workforce.workforce_id, WorkforceAssignment.state))).all()

    per: dict[str, dict] = {}
    for name, wid, state, n in rows:
        d = per.setdefault(name, {"workforce_id": str(wid), **dict.fromkeys(STATES, 0)})
        if state:
            d[state] = d.get(state, 0) + int(n)

    out = {}
    for name, d in sorted(per.items()):
        decided = d["accepted"] + d["rejected"]
        ci = wilson_interval(d["accepted"], decided, confidence)
        proven = decided >= MIN_BATCHES_FOR_RATING
        out[name] = {
            **d, "decided": decided, "accept_rate": ci, "proven": proven,
            "routing_weight": ci["lo"] if proven else UNPROVEN_WORKFORCE_WEIGHT,
            "note": None if proven else (
                f"only {decided} decided batches: this workforce is unproven, not poor"),
        }
    return {"per_workforce": out}


async def route_job(db: AsyncSession, *, job_id: str, required_classes: list[str] | None = None) -> dict:
    """Pick the workforce for a job: capable, active, under capacity, best proven.

    Capability first, because a workforce that cannot label the classes in a job is not a cheap option, it
    is a rejected batch and a wasted week. Capacity second, because overloading the best vendor is how a
    quality rating decays. Rating last, and it only breaks the tie among workforces that pass the first two.
    """
    job = await db.get(LabelJob, _uuid.UUID(job_id))
    if job is None:
        raise WorkforceError("job not found")

    ratings = (await workforce_rating(db))["per_workforce"]
    today = datetime.now(UTC).date()

    candidates = []
    for wf in (await db.execute(select(Workforce).where(Workforce.active.is_(True)))).scalars().all():
        classes = set((wf.capabilities or {}).get("classes") or [])
        if required_classes and not set(required_classes).issubset(classes):
            # An undeclared capability is not a wildcard. Reading an empty list as "can label anything"
            # would route 3D cuboid work to a vendor who never claimed to do it, and the first anybody
            # hears of it is a rejected batch a week later. A workforce that declares nothing can still
            # take jobs that ask for nothing in particular.
            continue
        if wf.capacity_jobs_per_day:
            used = (await db.execute(
                select(func.count(WorkforceAssignment.assignment_id))
                .where(WorkforceAssignment.workforce_id == wf.workforce_id,
                       func.date(WorkforceAssignment.dispatched_at) == today))).scalar() or 0
            if used >= wf.capacity_jobs_per_day:
                continue
        r = ratings.get(wf.name, {})
        candidates.append((r.get("routing_weight", UNPROVEN_WORKFORCE_WEIGHT), wf))

    if not candidates:
        return {"workforce_id": None,
                "reason": "no active workforce is capable of these classes and under capacity today"}

    candidates.sort(key=lambda t: (-t[0], t[1].name))    # name breaks ties so routing is deterministic
    weight, chosen = candidates[0]
    return {"workforce_id": str(chosen.workforce_id), "name": chosen.name,
            "routing_weight": round(weight, 4), "considered": len(candidates)}


def verify_return_signature(secret: str, body: bytes, signature: str) -> bool:
    """Check a workforce callback, using the webhook verifier so the replay window applies here too."""
    from services.integrations.webhooks import verify

    return verify(secret, body, signature)


async def get_workforce(db: AsyncSession, workforce_id: str) -> Workforce | None:
    """A workforce by id, tolerating a malformed id rather than raising.

    The caller is an unauthenticated outside party at this point, so a bad id is an ordinary wrong request
    and should get the same answer as an unknown one, not a 500 that confirms the id was well-formed.
    """
    try:
        return await db.get(Workforce, _uuid.UUID(workforce_id))
    except (ValueError, AttributeError):
        return None


async def get_assignment(db: AsyncSession, assignment_id: str) -> WorkforceAssignment | None:
    try:
        return await db.get(WorkforceAssignment, _uuid.UUID(assignment_id))
    except (ValueError, AttributeError):
        return None
