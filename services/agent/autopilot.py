"""The overnight autopilot (#18): label the new drives while the team sleeps, and hand people only what it is
unsure of.

Each night, outside the office-hours quiet window (`phase4.govern.quiet_hours_ist`) and after the start time a
person set, one GPU tool job (`autopilot_night`, run by the GPU worker under the single GPU slot) chains:

1. **Auto-label** the sessions nobody has labelled yet: real sessions with frames, no labels, no finished or
   queued auto-label job, and not held back by the inspector's health gate. Each one runs through the
   ordinary auto-label job (`services.autolabel.jobs.run_autolabel_job`), so its gate, its kill-switch
   wiring and its project class set are the ones a person pressing the button gets.
2. **VLM check** (Path C, `VlmVerifier`) of the labels the gate proposed to auto-accept, least confident
   first, within a per-night cap and the GPU budget.
3. **Error sweep**: the label-error detectors (services/errordetect) over the sessions labelled tonight.
4. **Auto-accept only where the autonomy ladder allows** (`services/govern/class_autonomy.py`): a proposed
   auto-accept stands only if its class is at level 1 or above, the governance switch allows auto-accept,
   the VLM agreed, and no detector flagged it. Everything else goes to the review queue.

Every label it sends to review is stamped with one AgentRun, so the night is one revert away, and the
morning report (which classes were accepted, which escalated and why, GPU minutes, what was left for
tomorrow) is written onto the night's row and shown on /autopilot. It fails closed: a label the VLM could not
check tonight (budget spent, model down, kill switch, quiet hours starting) goes to review, not to accepted.

A dry run (`plan_night`) reads the same state and reports what a night would do, changing nothing.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime

from sqlalchemy import exists, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, AutolabelJob, AutopilotConfig, AutopilotNight, Frame, Object
from db.models import Session as DbSession
from db.models_tools_agents import DEFAULT_STEPS
from services.tools.registry import IST, _parse_window, in_quiet_hours

log = get_logger("agent.autopilot")

KIND = "autopilot"                 # the AgentRun kind of a night's review routing
TOOL_KIND = "autopilot_night"      # the GPU tool job that runs a night
EXECUTOR = "autopilot"             # AutolabelJob.params.executor: neither the API nor the worker claims these
STEPS = tuple(DEFAULT_STEPS)
# Label-error detectors run per session. embedding_outlier is left out: tonight's sessions have no embeddings
# yet (the embed daemon reaches them later), so it could only ever find nothing.
DETECTORS = ("confident_learning", "track_inconsistent", "cross_cam_inconsistent", "critic_flag",
             "near_dup_inconsistent", "policy_violation")
# What a session costs to auto-label when no finished job on this box says otherwise. Replaced by the measured
# rate as soon as there is one; the plan says which it used.
FALLBACK_SEC_PER_FRAME = 6.0
FALLBACK_SEC_PER_VLM = 2.0

LIMITS = {"gpu_minutes": (5, 720), "max_sessions": (1, 500), "vlm_max_checks": (0, 5000)}


class AlreadyRan(RuntimeError):
    """A real night already exists for this calendar night."""


# ---- settings -------------------------------------------------------------------------------------------------

async def get_config(db: AsyncSession) -> AutopilotConfig:
    """The single settings row, created with the defaults (off) on first read."""
    cfg = await db.get(AutopilotConfig, 1)
    if cfg is None:
        cfg = AutopilotConfig(id=1, enabled=False, killed=False, gpu_minutes=240, max_sessions=20,
                              vlm_max_checks=300, earliest_start_ist="20:00", steps=dict(DEFAULT_STEPS))
        db.add(cfg)
        try:
            await db.commit()
            await db.refresh(cfg)
        except IntegrityError:        # another request created it first
            await db.rollback()
            cfg = await db.get(AutopilotConfig, 1)
            assert cfg is not None
    return cfg


def steps_of(cfg_or_dict) -> dict[str, bool]:
    raw = cfg_or_dict.steps if isinstance(cfg_or_dict, AutopilotConfig) else (cfg_or_dict or {}).get("steps")
    return {k: bool((raw or {}).get(k, v)) for k, v in DEFAULT_STEPS.items()}


def config_dict(cfg: AutopilotConfig) -> dict:
    return {"enabled": bool(cfg.enabled), "killed": bool(cfg.killed), "killed_reason": cfg.killed_reason,
            "killed_at": cfg.killed_at.isoformat() if cfg.killed_at else None,
            "gpu_minutes": int(cfg.gpu_minutes), "max_sessions": int(cfg.max_sessions),
            "vlm_max_checks": int(cfg.vlm_max_checks), "earliest_start_ist": cfg.earliest_start_ist,
            "steps": steps_of(cfg), "updated_at": cfg.updated_at.isoformat() if cfg.updated_at else None}


def _hhmm(spec: str) -> dtime:
    h, m = (int(x) for x in spec.strip().split(":"))
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"{spec!r} is not a time of day")
    return dtime(h, m)


def validate_settings(body: dict) -> dict:
    """Clean a settings update; raises ValueError with a sentence a person can act on."""
    out: dict = {}
    if "enabled" in body and body["enabled"] is not None:
        out["enabled"] = bool(body["enabled"])
    for key, (lo, hi) in LIMITS.items():
        if body.get(key) is not None:
            v = int(body[key])
            if not lo <= v <= hi:
                raise ValueError(f"{key.replace('_', ' ')} must be between {lo} and {hi}")
            out[key] = v
    if body.get("earliest_start_ist"):
        _hhmm(str(body["earliest_start_ist"]))
        out["earliest_start_ist"] = str(body["earliest_start_ist"]).strip()
    if body.get("steps") is not None:
        unknown = set(body["steps"]) - set(STEPS)
        if unknown:
            raise ValueError(f"unknown step(s) {sorted(unknown)}; the steps are {list(STEPS)}")
        out["steps"] = {k: bool(body["steps"].get(k, v)) for k, v in DEFAULT_STEPS.items()}
    return out


# ---- the night window -----------------------------------------------------------------------------------------

def night_of(now: datetime | None = None) -> date:
    """The calendar night a moment belongs to: an evening and the morning after it are one night."""
    return ((now or datetime.now(UTC)).astimezone(IST) - timedelta(hours=12)).date()


def window_state(cfg: AutopilotConfig, now: datetime | None = None) -> dict:
    """Whether a night may run now, and when it next can. The window opens at the start time a person set
    and closes when the office-hours quiet window begins."""
    from core.config import get_settings

    now = now or datetime.now(UTC)
    spec = get_settings().phase4.govern.quiet_hours_ist
    quiet = in_quiet_hours(now, spec)
    win = _parse_window(spec)
    t = now.astimezone(IST).time()
    earliest = _hhmm(cfg.earliest_start_ist)
    morning_end = win[0] if win else dtime(6, 0)
    at_night = t >= earliest or t < morning_end
    open_ = at_night and not quiet
    next_open = None
    if not open_:
        # Walk forward in quarter hours with the same rule: simpler to trust than solving the two windows.
        probe = now.astimezone(IST).replace(second=0, microsecond=0)
        for _ in range(4 * 48):
            probe += timedelta(minutes=15)
            pt = probe.time()
            if (pt >= earliest or pt < morning_end) and not in_quiet_hours(probe, spec):
                next_open = probe.isoformat()
                break
    reason = None
    if quiet:
        reason = f"office hours ({spec} IST): nothing that auto-labels starts now"
    elif not at_night:
        reason = f"the night starts at {cfg.earliest_start_ist} IST"
    return {"open": open_, "quiet": quiet, "quiet_hours_ist": spec, "earliest_start_ist": cfg.earliest_start_ist,
            "closes_at_ist": morning_end.strftime("%H:%M"), "reason": reason,
            "next_open": next_open, "night": night_of(now).isoformat()}


# ---- measuring what a night costs -----------------------------------------------------------------------------

async def sec_per_frame(db: AsyncSession) -> tuple[float, str]:
    """Seconds of GPU slot per auto-labelled frame, from the last finished jobs on this box."""
    rows = (await db.execute(
        select(AutolabelJob.counts, AutolabelJob.created_at, AutolabelJob.updated_at)
        .where(AutolabelJob.status == "done").order_by(AutolabelJob.updated_at.desc()).limit(20))).all()
    secs = frames = 0.0
    for counts, a, b in rows:
        n = (counts or {}).get("frames") or (counts or {}).get("n_frames") or 0
        if n and a and b and b > a:
            secs += (b - a).total_seconds()
            frames += float(n)
    if frames >= 10:
        return round(secs / frames, 2), f"measured over the last {len(rows)} finished auto-label jobs"
    return FALLBACK_SEC_PER_FRAME, "no finished auto-label jobs to measure; a planning constant"


async def sec_per_vlm(db: AsyncSession) -> tuple[float, str]:
    """Seconds per VLM check, as the last real night measured it."""
    last = (await db.execute(select(AutopilotNight.report).where(
        AutopilotNight.dry_run.is_(False), AutopilotNight.status.in_(("done", "stopped")))
        .order_by(AutopilotNight.created_at.desc()).limit(5))).scalars().all()
    for rep in last:
        v = (rep or {}).get("vlm") or {}
        if v.get("checked") and v.get("seconds"):
            return round(float(v["seconds"]) / float(v["checked"]), 2), "measured by the last night"
    return FALLBACK_SEC_PER_VLM, "no night has measured it yet; a planning constant"


async def candidate_sessions(db: AsyncSession, limit: int = 500) -> list[dict]:
    """Sessions not yet labelled: real, with frames, no labels at all, and no auto-label job done or queued.
    Oldest first, so a backlog drains in the order it arrived."""
    from services.inspector.health import is_gated

    n_frames = (select(func.count()).select_from(Frame).where(Frame.session_id == DbSession.session_id)
                .correlate(DbSession).scalar_subquery())
    has_obj = exists().where(Frame.session_id == DbSession.session_id, Object.frame_id == Frame.frame_id)
    has_job = exists().where(AutolabelJob.session_id == DbSession.session_id,
                             AutolabelJob.status.in_(("done", "pending", "running")))
    rows = (await db.execute(
        select(DbSession.session_id, DbSession.vehicle_id, DbSession.city, DbSession.created_at, n_frames.label("n"))
        .where(DbSession.origin == "real", ~has_obj, ~has_job, n_frames > 0)
        .order_by(DbSession.created_at).limit(limit))).all()
    out = []
    for sid, vehicle, city, created, n in rows:
        out.append({"session_id": str(sid), "vehicle": vehicle, "city": city, "frames": int(n),
                    "created_at": created.isoformat() if created else None,
                    "gated": await is_gated(db, sid)})
    return out


async def ladder_view(db: AsyncSession) -> dict:
    """The classes the ladder lets auto-accept tonight (level 1+), with their measured thresholds."""
    from services.govern.class_autonomy import ladder_snapshot
    from services.govern.registry import get_champion
    from services.oraclyx.threshold_fit import active_thresholds

    snap = await ladder_snapshot(db)
    champ = await get_champion(db, "detection")
    fitted = (await active_thresholds(db, champ.model_version)).get("by_class") if champ else {}
    allowed = [{"class_name": r["class_name"], "class_id": r["class_id"], "level": r["level"],
                "threshold": (fitted or {}).get(r["class_id"]), "set_by": r["set_by"]}
               for r in snap if r["level"] >= 1]
    return {"allowed": allowed, "propose_only": sum(1 for r in snap if r["level"] < 1),
            "champion": champ.model_version if champ else None, "levels": {r["class_id"]: r["level"] for r in snap}}


# ---- the plan (a dry run is exactly this) ----------------------------------------------------------------------

async def plan_night(db: AsyncSession, cfg: AutopilotConfig | None = None, *, now: datetime | None = None) -> dict:
    """What a night would do if it started now. Reads only; commits nothing."""
    from core.config import get_settings
    from services.govern.killswitch import get_state

    cfg = cfg or await get_config(db)
    now = now or datetime.now(UTC)
    steps = steps_of(cfg)
    window = window_state(cfg, now)
    gov = await get_state(db)
    spf, spf_basis = await sec_per_frame(db)
    spv, spv_basis = await sec_per_vlm(db)
    budget_s = int(cfg.gpu_minutes) * 60

    cands = await candidate_sessions(db)
    would: list[dict] = []
    over_budget: list[dict] = []
    over_count: list[dict] = []
    gated: list[dict] = []
    vlm_reserve = spv * int(cfg.vlm_max_checks) if steps["vlm_check"] else 0.0
    used = 0.0
    for c in cands:
        if c["gated"]:
            gated.append(c)
            continue
        est = round(c["frames"] * spf, 1)
        c = {**c, "est_seconds": est}
        if not steps["autolabel"]:
            over_count.append(c)
        elif len(would) >= int(cfg.max_sessions):
            over_count.append(c)
        elif used + est + vlm_reserve > budget_s and would:
            over_budget.append(c)
        else:
            would.append(c)
            used += est

    ladder = await ladder_view(db)
    already = (await db.execute(select(AutopilotNight.night_id, AutopilotNight.status).where(
        AutopilotNight.night_date == night_of(now), AutopilotNight.dry_run.is_(False)))).first()

    blockers = []
    if not cfg.enabled:
        blockers.append("the autopilot is switched off")
    if cfg.killed:
        blockers.append(f"the kill switch is on ({cfg.killed_reason or 'no reason given'})")
    if not gov.loop_enabled:
        blockers.append(f"the governance loop is paused ({gov.paused_reason or 'no reason given'})")
    if not window["open"]:
        blockers.append(window["reason"])
    if already is not None:
        blockers.append(f"tonight already ran ({already[1]})")

    frames = sum(c["frames"] for c in would)
    vlm_cfg = get_settings().models.vlm
    actions = []
    if steps["autolabel"]:
        actions.append(f"auto-label {len(would)} session(s), {frames:,} frames, about {round(used / 60, 1)} GPU minutes"
                       if would else "auto-label nothing: every session with frames already has labels or a job")
        if over_budget:
            actions.append(f"leave {len(over_budget)} session(s) for a later night: the GPU budget would run out")
        if over_count:
            actions.append(f"leave {len(over_count)} session(s) for a later night: at most {cfg.max_sessions} a night")
        if gated:
            actions.append(f"skip {len(gated)} session(s) that failed the inspector's health checks")
    else:
        actions.append("auto-labelling is switched off; only labels it creates are checked, so nothing would run")
    if steps["vlm_check"]:
        actions.append(f"ask the VLM about up to {cfg.vlm_max_checks} proposed auto-accepts, least confident first"
                       + ("" if vlm_cfg.enabled else " (the VLM is disabled in settings, so they would go to review)"))
    if steps["error_sweep"]:
        actions.append(f"run {len(DETECTORS)} label-error detectors over the sessions it labels")
    if steps["auto_accept"] and gov.auto_accept_enabled:
        names = ", ".join(a["class_name"] for a in ladder["allowed"][:12])
        actions.append(f"keep auto-accepts only for {len(ladder['allowed'])} class(es) the ladder allows"
                       + (f" ({names}{', ...' if len(ladder['allowed']) > 12 else ''})" if names else "")
                       + "; send every other label to review")
    else:
        actions.append("auto-accept nothing tonight ("
                       + ("the step is off" if not steps["auto_accept"] else "governance holds auto-accept off")
                       + "); every label goes to review")
    actions.append("write a morning report")

    return {
        "at": now.isoformat(), "night": night_of(now).isoformat(), "window": window, "steps": steps,
        "settings": config_dict(cfg),
        "governance": {"loop_enabled": bool(gov.loop_enabled), "auto_accept_enabled": bool(gov.auto_accept_enabled)},
        "would_start": not blockers and bool(would or not steps["autolabel"]),
        "blockers": blockers, "actions": actions,
        "sessions": {"would_label": would, "over_budget": over_budget, "over_count": over_count, "gated": gated,
                     "candidates": len(cands)},
        "estimate": {"gpu_minutes": round((used + (vlm_reserve if would else 0)) / 60, 1),
                     "autolabel_minutes": round(used / 60, 1),
                     "vlm_minutes": round((vlm_reserve if would else 0) / 60, 1),
                     "budget_minutes": int(cfg.gpu_minutes), "sec_per_frame": spf, "sec_per_frame_basis": spf_basis,
                     "sec_per_vlm_check": spv, "sec_per_vlm_basis": spv_basis},
        "ladder": {"allowed": ladder["allowed"], "propose_only": ladder["propose_only"], "champion": ladder["champion"]},
        "vlm": {"enabled": bool(vlm_cfg.enabled)},
    }


async def record_dry_run(db: AsyncSession, *, user_id: uuid.UUID | None = None, now: datetime | None = None) -> AutopilotNight:
    """Plan a night and keep the plan in the history, marked as a dry run. Changes no labels."""
    cfg = await get_config(db)
    plan = await plan_night(db, cfg, now=now)
    n = AutopilotNight(night_date=night_of(now), dry_run=True, status="done", trigger="dry_run",
                       settings=config_dict(cfg), plan=plan, report={"dry_run": True, "actions": plan["actions"]},
                       created_by=user_id, started_at=datetime.now(UTC), finished_at=datetime.now(UTC))
    db.add(n)
    await db.commit()
    await db.refresh(n)
    return n


# ---- starting a night -----------------------------------------------------------------------------------------

async def start_night(db: AsyncSession, *, trigger: str, user_id: uuid.UUID | None = None,
                      now: datetime | None = None) -> AutopilotNight:
    """Record tonight and queue its GPU job. Raises AlreadyRan when a real night exists for this night."""
    from services.tools.registry import enqueue

    cfg = await get_config(db)
    plan = await plan_night(db, cfg, now=now)
    night = AutopilotNight(night_date=night_of(now), dry_run=False, status="planned", trigger=trigger,
                           settings=config_dict(cfg), plan=plan, report={}, created_by=user_id)
    db.add(night)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise AlreadyRan(f"the night of {night_of(now)} already has a run") from None
    job = await enqueue(db, TOOL_KIND, {"night_id": str(night.night_id)}, user_id=user_id)
    night.tool_job_id = job.job_id
    await db.commit()
    await db.refresh(night)
    log.info("autopilot.night_queued", night_id=str(night.night_id), job_id=str(job.job_id), trigger=trigger,
             sessions=len(plan["sessions"]["would_label"]))
    return night


async def maybe_autopilot(db: AsyncSession, now: datetime | None = None) -> dict:
    """Scheduler hook (services/agent/runtime/schedule.py): start tonight's run once, when it may."""
    from services.govern.killswitch import get_state

    now = now or datetime.now(UTC)
    cfg = await get_config(db)
    if not cfg.enabled:
        return {"ran": False, "reason": "switched off"}
    if cfg.killed:
        return {"ran": False, "reason": "kill switch on"}
    window = window_state(cfg, now)
    if not window["open"]:
        return {"ran": False, "reason": window["reason"]}
    if not (await get_state(db)).loop_enabled:
        return {"ran": False, "reason": "governance loop paused"}
    if (await db.execute(select(AutopilotNight.night_id).where(
            AutopilotNight.night_date == night_of(now), AutopilotNight.dry_run.is_(False)))).first() is not None:
        return {"ran": False, "reason": "already ran tonight"}
    try:
        night = await start_night(db, trigger="scheduler", now=now)
    except AlreadyRan as exc:
        return {"ran": False, "reason": str(exc)}
    return {"ran": True, "night_id": str(night.night_id), "job_id": str(night.tool_job_id)}


async def kill(db: AsyncSession, *, reason: str, user_id: uuid.UUID | None = None) -> dict:
    """Stop now: set the kill switch, cancel tonight's queued or running job and its current auto-label job.
    A running night notices at its next step and sends what it had not checked to review."""
    from db.models import ToolJob
    from services.job_control import cancel_job

    cfg = await get_config(db)
    cfg.killed, cfg.killed_reason, cfg.killed_at, cfg.updated_by = True, reason[:500], datetime.now(UTC), user_id
    await db.commit()
    await db.refresh(cfg)
    stopped = []
    nights = (await db.execute(select(AutopilotNight).where(
        AutopilotNight.dry_run.is_(False), AutopilotNight.status.in_(("planned", "running"))))).scalars().all()
    for n in nights:
        if n.tool_job_id:
            job = await db.get(ToolJob, n.tool_job_id)
            if job is not None and job.status in ("pending", "running"):
                job.status = "canceled"
                if n.status == "planned":
                    n.status, n.stopped_reason, n.finished_at = "stopped", "kill switch before it started", datetime.now(UTC)
                stopped.append(str(n.night_id))
        cur = (n.report or {}).get("current_autolabel_job")
        if cur:
            await cancel_job(db, AutolabelJob, uuid.UUID(cur))
    await db.commit()
    log.warning("autopilot.killed", reason=reason, nights=stopped)
    return {"killed": True, "reason": reason, "stopped_nights": stopped}


async def rearm(db: AsyncSession, *, user_id: uuid.UUID | None = None) -> AutopilotConfig:
    cfg = await get_config(db)
    cfg.killed, cfg.killed_reason, cfg.killed_at, cfg.updated_by = False, None, None, user_id
    await db.commit()
    await db.refresh(cfg)
    return cfg


# ---- running a night (inside the GPU worker) ------------------------------------------------------------------

async def _persist_candidates(db: AsyncSession, found: list[dict]) -> int:
    """Store detector hits, replacing only pending candidates for the same object and detector, so a
    per-session sweep never clears another session's queue (run_detection clears corpus-wide)."""
    from sqlalchemy import delete, tuple_

    from db.models import ErrorCandidate

    best: dict = {}
    for c in found:
        key = (c["object_id"], c["kind"])
        if key not in best or c["score"] > best[key]["score"]:
            best[key] = c
    if not best:
        return 0
    keys = [(uuid.UUID(o), k) for o, k in best]
    for i in range(0, len(keys), 500):
        await db.execute(delete(ErrorCandidate).where(
            tuple_(ErrorCandidate.object_id, ErrorCandidate.kind).in_(keys[i:i + 500]),
            ErrorCandidate.status == "pending"))
    for c in best.values():
        db.add(ErrorCandidate(object_id=uuid.UUID(c["object_id"]), kind=c["kind"], score=c["score"],
                              proposed_label=c.get("proposed_label"), detail=c.get("detail", {}), status="pending"))
    await db.commit()
    return len(best)


async def error_sweep(db: AsyncSession, session_ids: list[str]) -> dict:
    """The label-error detectors over the given sessions. Returns {flagged: {object_id: [kinds]}, by_kind, errors}."""
    from services.errordetect.confident import detect_confident_learning
    from services.errordetect.consistency import detect_consistency
    from services.errordetect.critic_detector import detect_critic
    from services.errordetect.near_dup import detect_near_dup_inconsistent
    from services.errordetect.policy import detect_policy_violations

    flagged: dict[str, list[str]] = {}
    by_kind: Counter = Counter()
    errors: list[str] = []
    for sid in session_ids:
        found: list[dict] = []
        for name, fn in (("confident_learning", detect_confident_learning), ("consistency", detect_consistency),
                         ("critic_flag", detect_critic), ("near_dup_inconsistent", detect_near_dup_inconsistent),
                         ("policy_violation", detect_policy_violations)):
            try:
                found += [c for c in await fn(db, sid) if c["kind"] in DETECTORS]
            except Exception as exc:  # noqa: BLE001 - one detector failing is a line in the report, not a stop
                await db.rollback()
                errors.append(f"{name} on {sid[:8]}: {str(exc)[:160]}")
        await _persist_candidates(db, found)
        for c in found:
            flagged.setdefault(c["object_id"], [])
            if c["kind"] not in flagged[c["object_id"]]:
                flagged[c["object_id"]].append(c["kind"])
            by_kind[c["kind"]] += 1
    return {"flagged": flagged, "by_kind": dict(by_kind), "errors": errors}


class _Night:
    """The running state of one night: its budget clock and the reasons to stop."""

    def __init__(self, night_id: uuid.UUID, settings: dict, progress, clock=time.monotonic):
        self.night_id = night_id
        self.settings = settings
        self.progress = progress
        self.clock = clock
        self.t0 = clock()
        self.budget_s = float(settings.get("gpu_minutes", 240)) * 60
        self.canceled = False

    def elapsed(self) -> float:
        return self.clock() - self.t0

    async def stop_reason(self) -> str | None:
        from db.session import get_sessionmaker

        if self.canceled:
            return "the job was cancelled"
        async with get_sessionmaker()() as db:
            cfg = await db.get(AutopilotConfig, 1)
        if cfg is not None and cfg.killed:
            return "kill switch"
        if in_quiet_hours():
            return "office hours began"
        if self.elapsed() >= self.budget_s:
            return "GPU budget spent"
        return None

    async def step(self, frac: float, stage: str) -> None:
        try:
            await self.progress(frac, stage)
        except Exception as exc:  # noqa: BLE001 - JobCanceled: finish safely rather than abandon mid-night
            if type(exc).__name__ == "JobCanceled":
                self.canceled = True
            else:
                raise


async def _update_night(night_id: uuid.UUID, **fields) -> None:
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        n = await db.get(AutopilotNight, night_id)
        if n is not None:
            for k, v in fields.items():
                setattr(n, k, v)
            await db.commit()


async def _autolabel_one(sid: str) -> uuid.UUID:
    from db.session import get_sessionmaker
    from services.autolabel.jobs import job_params

    async with get_sessionmaker()() as db:
        job = AutolabelJob(job_id=uuid.uuid4(), session_id=uuid.UUID(sid), status="pending",
                           params=job_params(None, True, EXECUTOR))
        db.add(job)
        await db.commit()
        return job.job_id


async def run_night(night_id: uuid.UUID, progress, *, autolabel_fn=None, verifier=None) -> dict:
    """Run one night to its report. `progress(frac, stage)` is the tool job's; `autolabel_fn(job_id)` and
    `verifier` are seams for tests (default: the real auto-label job and the configured Path C VLM)."""
    from core.config import get_settings
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology

    maker = get_sessionmaker()
    onto = get_ontology()
    if autolabel_fn is None:
        from services.autolabel.jobs import run_autolabel_job as autolabel_fn
    started = datetime.now(UTC)
    run_id = uuid.uuid4()
    async with maker() as db:
        night = await db.get(AutopilotNight, night_id)
        if night is None:
            raise LookupError(f"autopilot night {night_id} not found")
        if night.status not in ("planned", "running"):
            return {"skipped": f"night is {night.status}"}
        cfg = await get_config(db)
        settings = dict(night.settings or config_dict(cfg))
        steps = steps_of(settings)
        db.add(AgentRun(run_id=run_id, kind=KIND, scope={"night_id": str(night_id)}, status="running",
                        policy=settings, counts={}, changes={}, critic={}, created_by="autopilot"))
        night.agent_run_id, night.status, night.started_at = run_id, "running", started
        await db.commit()
        plan = await plan_night(db, cfg)

    st = _Night(night_id, settings, progress)
    report: dict = {"steps": steps, "sessions": [], "notes": [], "budget_minutes": int(st.budget_s // 60)}
    stop: str | None = None
    labelled: list[str] = []

    # 1. auto-label
    todo = plan["sessions"]["would_label"] if steps["autolabel"] else []
    report["left_for_later"] = len(plan["sessions"]["over_budget"]) + len(plan["sessions"]["over_count"])
    label_s = 0.0
    for i, s in enumerate(todo):
        stop = await st.stop_reason()
        if stop:
            report["left_for_later"] += len(todo) - i
            break
        if st.elapsed() + float(s["est_seconds"]) > st.budget_s and labelled:
            report["sessions"].append({**s, "status": "left", "detail": "would exceed tonight's GPU budget"})
            report["left_for_later"] += 1
            continue
        job_id = await _autolabel_one(s["session_id"])
        report["current_autolabel_job"] = str(job_id)
        await _update_night(night_id, report=dict(report))
        await st.step(0.02 + 0.5 * i / max(1, len(todo)), f"auto-labelling {i + 1} of {len(todo)}")
        t = st.clock()
        await autolabel_fn(job_id)
        dt = st.clock() - t
        label_s += dt
        async with maker() as db:
            job = await db.get(AutolabelJob, job_id)
            status, counts, err = (job.status, dict(job.counts or {}), job.error) if job else ("missing", {}, None)
        report["sessions"].append({"session_id": s["session_id"], "vehicle": s.get("vehicle"), "city": s.get("city"),
                                   "frames": s["frames"], "status": status, "objects": counts.get("objects"),
                                   "by_state": counts.get("by_state"), "seconds": round(dt, 1),
                                   "error": (err or "")[:300] or None, "autolabel_job": str(job_id)})
        report.pop("current_autolabel_job", None)
        if status == "done":
            labelled.append(s["session_id"])
    report["labelled_sessions"] = len(labelled)
    report["autolabel_seconds"] = round(label_s, 1)
    if todo:
        try:   # hand the card back before the VLM: the detectors are done for the night
            from core.model_manager import empty_cuda_cache, manager

            manager().release_all()
            empty_cuda_cache()
        except Exception:  # noqa: BLE001
            pass

    # Tonight's labels and the gate's proposed auto-accepts among them.
    cand: list = []
    tonight: Counter = Counter()
    if labelled:
        sids = [uuid.UUID(x) for x in labelled]
        async with maker() as db:
            for state, cid, n in (await db.execute(
                    select(Object.state, Object.class_id, func.count()).join(Frame, Frame.frame_id == Object.frame_id)
                    .where(Frame.session_id.in_(sids), Object.created_at >= started)
                    .group_by(Object.state, Object.class_id))).all():
                tonight[(state, int(cid))] += int(n)
            cand = list((await db.execute(
                select(Object.object_id, Object.class_id, Object.conf, Object.bbox, Frame.img_uri, Frame.frame_id)
                .join(Frame, Frame.frame_id == Object.frame_id)
                .where(Frame.session_id.in_(sids), Object.created_at >= started, Object.state == "auto_accept",
                       Object.source != "human")
                .order_by(Object.conf))).all())
    report["labels_created"] = sum(tonight.values())
    report["proposed_auto_accepts"] = len(cand)

    demote: dict[str, str] = {}         # object_id -> why it goes to review
    async with maker() as db:
        from services.govern.killswitch import get_state

        gov = await get_state(db)
        ladder = await ladder_view(db)
    levels = ladder["levels"]
    allowed = []
    for row in cand:
        oid = str(row.object_id)
        if not steps["auto_accept"]:
            demote[oid] = "auto-accept step is off"
        elif not gov.auto_accept_enabled:
            demote[oid] = "governance holds auto-accept off"
        elif levels.get(int(row.class_id), 0) < 1:
            demote[oid] = "ladder: this class is propose-only"
        else:
            allowed.append(row)

    # 2. VLM check of what the ladder would let through
    vlm: dict = {"checked": 0, "disagreed": 0, "errors": 0, "seconds": 0.0}
    checked_ok: set[str] = set()
    if allowed and steps["vlm_check"] and not stop:
        cap = int(settings.get("vlm_max_checks", 300))
        if verifier is None and get_settings().models.vlm.enabled:
            try:
                from services.autolabel.paths.path_c_vlm import VlmVerifier, make_vlm_client

                verifier = VlmVerifier(make_vlm_client(get_settings()), onto)
            except Exception as exc:  # noqa: BLE001
                vlm["unavailable"] = str(exc)[:200]
        if verifier is None and "unavailable" not in vlm:
            vlm["unavailable"] = "the VLM is disabled in settings"
        if verifier is not None:
            from core.storage import get_object_store
            from services.recall.backends import load_image_bgr

            store = get_object_store()
            img_cache: dict = {}
            consecutive = 0
            t = st.clock()
            for j, row in enumerate(allowed):
                if vlm["checked"] >= cap:
                    break
                if j % 20 == 0:
                    stop = await st.stop_reason()
                    if stop:
                        break
                    await st.step(0.55 + 0.3 * j / max(1, min(cap, len(allowed))), f"VLM check {j} of {len(allowed)}")
                oid = str(row.object_id)
                try:
                    if row.frame_id not in img_cache:
                        img_cache.clear()
                        img_cache[row.frame_id] = await asyncio.to_thread(load_image_bgr, store, row.img_uri)
                    res = await asyncio.to_thread(verifier.verify_object, img_cache[row.frame_id],
                                                  tuple(float(x) for x in row.bbox), int(row.class_id))
                    consecutive = 0
                except Exception as exc:  # noqa: BLE001
                    vlm["errors"] += 1
                    consecutive += 1
                    if consecutive >= 5:
                        vlm["unavailable"] = f"the VLM failed 5 times in a row: {str(exc)[:160]}"
                        break
                    continue
                vlm["checked"] += 1
                cur = onto.by_id(int(row.class_id)).name
                if res.class_name and res.class_name != cur and res.confident:
                    vlm["disagreed"] += 1
                    demote[oid] = f"VLM thinks it is {res.class_name}"
                else:
                    checked_ok.add(oid)
            vlm["seconds"] = round(st.clock() - t, 1)
    report["vlm"] = vlm
    if steps["vlm_check"]:
        for row in allowed:
            oid = str(row.object_id)
            if oid not in checked_ok and oid not in demote:
                demote[oid] = "not checked by the VLM tonight"

    # 3. error sweep (CPU; runs even after a stop, since it only reads and files candidates)
    sweep: dict = {"flagged": {}, "by_kind": {}, "errors": []}
    if steps["error_sweep"] and labelled and stop != "kill switch" and not st.canceled:
        await st.step(0.88, "error sweep")
        async with maker() as db:
            sweep = await error_sweep(db, labelled)
        for row in allowed:
            oid = str(row.object_id)
            if oid in sweep["flagged"] and oid not in demote:
                demote[oid] = "error sweep: " + ", ".join(sweep["flagged"][oid])
    report["error_sweep"] = {"candidates": sum(sweep["by_kind"].values()), "by_kind": sweep["by_kind"],
                             "errors": sweep["errors"]}

    # 4. apply: everything not cleared goes to review, as one revertible run
    changes: dict = {}
    reasons: Counter = Counter()
    escalated_by_class: Counter = Counter()
    if demote:
        ids = [uuid.UUID(o) for o in demote]
        async with maker() as db:
            for k in range(0, len(ids), 500):
                for obj in (await db.execute(select(Object).where(Object.object_id.in_(ids[k:k + 500])))).scalars():
                    if obj.source == "human" or obj.state != "auto_accept":
                        continue          # somebody ruled while the night ran; theirs stands
                    why = demote[str(obj.object_id)]
                    changes[str(obj.object_id)] = {"from_state": obj.state, "from_source": obj.source}
                    obj.state = "review"
                    obj.version = (obj.version or 0) + 1
                    obj.provenance = {**(obj.provenance or {}), "agent_run_id": str(run_id),
                                      "autopilot": {"night_id": str(night_id), "reason": why}}
                    reasons[_reason_key(why)] += 1
                    escalated_by_class[onto.by_id(int(obj.class_id)).name] += 1
                await db.commit()
    kept: Counter = Counter()
    for row in cand:
        if str(row.object_id) not in changes and str(row.object_id) not in demote:
            kept[onto.by_id(int(row.class_id)).name] += 1
    for (state, cid), n in tonight.items():
        if state in ("review", "annotate"):
            escalated_by_class[onto.by_id(cid).name] += n

    report["auto_accepted"] = {"total": sum(kept.values()), "by_class": dict(kept.most_common())}
    report["sent_to_review"] = {"from_auto_accept": len(changes), "reasons": dict(reasons.most_common()),
                                "total": sum(escalated_by_class.values()),
                                "by_class": dict(escalated_by_class.most_common(25))}
    gpu_s = round(st.elapsed(), 1)
    report["gpu_minutes"] = round(gpu_s / 60, 1)
    report["stopped"] = stop or ("the job was cancelled" if st.canceled else None)
    report["notes"] = _narrate(report)
    status = "stopped" if report["stopped"] else "done"

    from services.agent.runtime.report import finish_run

    await finish_run(run_id, status="committed", report={k: report[k] for k in (
        "labelled_sessions", "labels_created", "proposed_auto_accepts", "gpu_minutes")} | {
        "auto_accepted": report["auto_accepted"]["total"], "sent_to_review": len(changes)},
        changes=changes, decision="autopilot_night")
    await _update_night(night_id, status=status, report=report, gpu_seconds=gpu_s, stopped_reason=report["stopped"],
                        finished_at=datetime.now(UTC))
    try:
        from services.notify import notify

        async with maker() as db:
            await notify(db, kind="nightly_digest", severity="warn" if report["stopped"] else "info",
                         title=f"autopilot: {report['labelled_sessions']} session(s) labelled, "
                               f"{report['auto_accepted']['total']} auto-accepted, "
                               f"{report['sent_to_review']['total']} for review",
                         body=" ".join(report["notes"][:3]), href="/autopilot",
                         subject_type="autopilot_night", subject_id=str(night_id))
    except Exception as exc:  # noqa: BLE001 - the report is on the row either way
        log.warning("autopilot.notify_failed", error=str(exc))
    log.info("autopilot.night_done", night_id=str(night_id), status=status, labelled=len(labelled),
             kept=sum(kept.values()), to_review=len(changes))
    return {"night_id": str(night_id), "status": status, "labelled_sessions": len(labelled),
            "auto_accepted": sum(kept.values()), "sent_to_review": len(changes), "gpu_minutes": report["gpu_minutes"]}


def _reason_key(why: str) -> str:
    """Group the per-label reasons for the report: 'VLM thinks it is bus' and '... car' are one line."""
    if why.startswith("VLM thinks"):
        return "VLM disagreed"
    if why.startswith("error sweep"):
        return "error sweep flagged it"
    return why


def _narrate(r: dict) -> list[str]:
    notes = []
    if r.get("labelled_sessions"):
        notes.append(f"Labelled {r['labelled_sessions']} new session(s): {r.get('labels_created', 0):,} labels.")
    else:
        notes.append("No new sessions were labelled.")
    aa = r.get("auto_accepted") or {}
    if aa.get("total"):
        top = ", ".join(f"{v} {k}" for k, v in list((aa.get("by_class") or {}).items())[:4])
        notes.append(f"Auto-accepted {aa['total']} label(s) the ladder allows ({top}).")
    sr = r.get("sent_to_review") or {}
    if sr.get("from_auto_accept"):
        why = ", ".join(f"{v} {k}" for k, v in list((sr.get("reasons") or {}).items())[:3])
        notes.append(f"Sent {sr['from_auto_accept']} proposed auto-accept(s) to review instead ({why}).")
    v = r.get("vlm") or {}
    if v.get("unavailable"):
        notes.append(f"VLM check did not run: {v['unavailable']}.")
    elif v.get("checked"):
        notes.append(f"The VLM checked {v['checked']} and disagreed with {v.get('disagreed', 0)}.")
    es = r.get("error_sweep") or {}
    if es.get("candidates"):
        notes.append(f"The error sweep filed {es['candidates']} likely error(s).")
    if r.get("left_for_later"):
        notes.append(f"{r['left_for_later']} session(s) left for a later night.")
    if r.get("stopped"):
        notes.append(f"Stopped early: {r['stopped']}.")
    notes.append(f"Used {r.get('gpu_minutes', 0)} of {r.get('budget_minutes')} GPU minutes.")
    return notes


def night_dict(n: AutopilotNight, *, full: bool = True) -> dict:
    out = {"night_id": str(n.night_id), "night_date": n.night_date.isoformat(), "dry_run": bool(n.dry_run),
           "status": n.status, "trigger": n.trigger, "tool_job_id": str(n.tool_job_id) if n.tool_job_id else None,
           "agent_run_id": str(n.agent_run_id) if n.agent_run_id else None, "gpu_minutes": round(n.gpu_seconds / 60, 1),
           "stopped_reason": n.stopped_reason,
           "created_at": n.created_at.isoformat() if n.created_at else None,
           "started_at": n.started_at.isoformat() if n.started_at else None,
           "finished_at": n.finished_at.isoformat() if n.finished_at else None,
           "notes": (n.report or {}).get("notes") or (n.report or {}).get("actions") or []}
    if full:
        out.update(plan=n.plan or {}, report=n.report or {}, settings=n.settings or {})
    else:
        r = n.report or {}
        out["summary"] = {"labelled_sessions": r.get("labelled_sessions"), "labels_created": r.get("labels_created"),
                          "auto_accepted": (r.get("auto_accepted") or {}).get("total"),
                          "sent_to_review": (r.get("sent_to_review") or {}).get("total"),
                          "would_label": len(((n.plan or {}).get("sessions") or {}).get("would_label") or [])}
    return out


__all__ = ["AlreadyRan", "KIND", "TOOL_KIND", "config_dict", "get_config", "kill", "maybe_autopilot", "night_dict",
           "night_of", "plan_night", "rearm", "record_dry_run", "run_night", "start_night", "validate_settings",
           "window_state"]
