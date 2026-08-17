"""Governance endpoints (M4.4): model registry + champion/challenger promotion, control-sample precision,
drift scan, the controller tick, the kill switch, and the audit log."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role
from services.govern import killswitch as K
from services.govern.audit import list_audit
from services.govern.champion import evaluate_and_promote
from services.govern.consent import export_consent_gate
from services.govern.control_sample import (
    measured_precision,
    record_verdict,
    seed_from_recent_auto_accepts,
)
from services.govern.controller import tick
from services.govern.cost import cost_gate, estimate_job_cost
from services.govern.drift import run_drift_scan
from services.govern.redaction_run import build_release_proof, governance_lineage, set_consent
from services.govern.registry import list_models, register, register_from_run

router = APIRouter()


# ---- M18 governance: redaction proof, consent/retention, cost ceilings, lineage ----
class RedactionProofIn(BaseModel):
    release_commit: str
    frame_ids: list[str]
    method_version: str = ""


@router.post("/govern/redaction/proof")
async def redaction_proof(payload: RedactionProofIn, db: AsyncSession = Depends(db_session)):
    """Build and sign a release redaction proof from the per-frame PII audits; a release with an unredacted
    frame fails the proof and names it. Rejects an oversized or malformed frame set with a 400."""
    try:
        return await build_release_proof(db, payload.release_commit, payload.frame_ids, payload.method_version)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class ConsentIn(BaseModel):
    session_id: uuid.UUID
    consent_status: str              # granted|denied|unknown
    legal_basis: str | None = None
    # The retention deadline. This field did not exist and the handler dropped the parameter, so every
    # consent record carried a null deadline, retention_status returned "retain" forever, and the retention
    # feature was unreachable end to end.
    retention_until: datetime | None = None


@router.post("/govern/consent")
async def consent(payload: ConsentIn, db: AsyncSession = Depends(db_session)):
    """Set a session's consent/retention record; export is refused later unless consent is 'granted'."""
    return await set_consent(db, payload.session_id, payload.consent_status, payload.legal_basis,
                             retention_until=payload.retention_until)


@router.get("/govern/retention/due")
async def retention_due(db: AsyncSession = Depends(db_session)):
    """Sessions past their retention deadline. Readable on its own so an operator can see what a sweep would
    erase before anything is deleted."""
    from services.govern.retention import expired_sessions

    return {"sessions": await expired_sessions(db)}


class RetentionSweepIn(BaseModel):
    # Defaults to a dry run: an irreversible bulk delete that runs the first time someone calls it to see
    # what it does is a trap.
    dry_run: bool = True
    limit: int = 50


@router.post("/govern/retention/sweep")
async def retention_sweep(payload: RetentionSweepIn, db: AsyncSession = Depends(db_session)):
    """Erase every session past its retention deadline, returning a certificate per erasure."""
    from services.govern.retention import run_retention_sweep

    return await run_retention_sweep(db, dry_run=payload.dry_run, limit=payload.limit)


class EraseIn(BaseModel):
    session_id: uuid.UUID
    reason: str = "data subject erasure request"
    dry_run: bool = True


@router.post("/govern/erase")
async def erase(payload: EraseIn, db: AsyncSession = Depends(db_session)):
    """Erase one session on request: frames, annotations, PII audits, and image blobs.

    This is the data-subject erasure path, which did not exist: the only deletion in the system was a
    per-object endpoint, nothing walked from a subject to its data, and nothing covered object storage, which
    database cascades do not reach. Returns a signed certificate, because a deletion that leaves no evidence
    it happened cannot be shown to a regulator.
    """
    from services.govern.retention import erase_session

    return await erase_session(db, payload.session_id, reason=payload.reason, dry_run=payload.dry_run)


@router.get("/govern/consent/{consent_status}/gate")
async def consent_gate(consent_status: str):
    """The export consent gate (fails closed on anything but 'granted')."""
    return export_consent_gate(consent_status)


class CostGateIn(BaseModel):
    gpu_hours: float
    hourly_usd: float
    spent_usd: float = 0.0
    per_job_cap_usd: float | None = None
    window_cap_usd: float | None = None


@router.post("/govern/cost/gate")
async def cost_ceiling(payload: CostGateIn):
    """Admit or refuse a cloud GPU job against the per-job cap and the remaining window budget, before it is
    dispatched."""
    from core.config import get_settings
    cfg = get_settings()
    est = estimate_job_cost(payload.gpu_hours, payload.hourly_usd)
    per_job = payload.per_job_cap_usd if payload.per_job_cap_usd is not None else cfg.cloud.per_job_cap_usd
    window = payload.window_cap_usd if payload.window_cap_usd is not None else cfg.cloud.budget_cap_usd
    return cost_gate(est, per_job, payload.spent_usd, window)


@router.get("/govern/lineage/{subject}")
async def lineage(subject: str, db: AsyncSession = Depends(db_session)):
    """The governance lineage of a subject: its audit decisions and (for a release) its redaction proof."""
    return await governance_lineage(db, subject)


# ---- registry + promotion ----
class RegisterIn(BaseModel):
    model_version: str
    task: str = "detection"
    gold_metrics: dict
    dataset_commit: str | None = None
    weights_uri: str | None = None
    notes: str | None = None


@router.post("/govern/registry/register")
async def registry_register(payload: RegisterIn, db: AsyncSession = Depends(db_session)):
    return await register(db, payload.model_version, payload.task, payload.gold_metrics,
                          payload.dataset_commit, payload.weights_uri, payload.notes)


@router.post("/govern/registry/register_run")
async def registry_register_run(run_id: str, task: str | None = None, db: AsyncSession = Depends(db_session)):
    return await register_from_run(db, run_id, task)


@router.get("/govern/registry")
async def registry_list(task: str | None = None, db: AsyncSession = Depends(db_session)):
    return await list_models(db, task)


@router.post("/govern/promote")
async def promote(model_version: str, task: str = "detection", db: AsyncSession = Depends(db_session)):
    """Attempt a promotion, and tell somebody how it went.

    A blocked promotion was the loop's quietest failure: the gate refused, the reason was recorded, and
    unless a reviewer happened to open this page it could sit unnoticed for a day while the flywheel idled.
    """
    from services.notify import notify

    result = await evaluate_and_promote(db, model_version, task)
    promoted = bool(result.get("promoted"))
    await notify(
        db, kind="model_promoted" if promoted else "promotion_blocked",
        severity="info" if promoted else "warn",
        title=(f"{task} model {model_version} promoted" if promoted
               else f"{task} promotion blocked: {model_version}"),
        body=result.get("reason") or result.get("detail"),
        href="/govern", subject_type="model", subject_id=model_version,
        meta={"task": task, "metrics": result.get("metrics") or {}})
    return result


# ---- control sample ----
# The control sample is the only measurement of whether the auto-accept gate is right, so a verdict on one
# is a governance act: anyone who could write these could move the number the corpus is judged by. Reading
# is annotator-level because judging them is work an annotator does.
@router.post("/govern/control/seed", dependencies=[Depends(require_role("reviewer"))])
async def control_seed(limit: int = 500, rate: float | None = None, db: AsyncSession = Depends(db_session)):
    return await seed_from_recent_auto_accepts(db, limit, rate)


class VerdictIn(BaseModel):
    verdict: str


@router.post("/govern/control/{sample_id}/verdict", dependencies=[Depends(require_role("reviewer"))])
async def control_verdict(sample_id: str, payload: VerdictIn, db: AsyncSession = Depends(db_session)):
    return await record_verdict(db, sample_id, payload.verdict)


@router.get("/govern/control/pending", dependencies=[Depends(require_role("annotator"))])
async def control_pending(limit: int = 100, db: AsyncSession = Depends(db_session)):
    """The control samples awaiting a verdict.

    This corpus holds 601 of them, every one unjudged, because there was no way to list what needed judging
    and the verdict route had no caller. Measured precision, which the drift detector watches and which is
    meant to be the number a buyer trusts over a self-reported one, has been null since the feature shipped.
    """
    from services.govern.control_sample import pending_samples

    return await pending_samples(db, limit)


@router.get("/govern/control/precision", dependencies=[Depends(require_role("annotator"))])
async def control_precision(db: AsyncSession = Depends(db_session)):
    return await measured_precision(db)


# ---- drift + controller ----
class DriftIn(BaseModel):
    ref_sessions: list[str] | None = None
    cur_sessions: list[str] | None = None


@router.post("/govern/drift/scan")
async def drift_scan(payload: DriftIn, db: AsyncSession = Depends(db_session)):
    """Scan for drift, and raise it if it breached.

    Superseded rather than appended: drift re-evaluates on a schedule, so an unresolved breach would
    otherwise produce a fresh line every cycle until the bell was pure noise.
    """
    from services.notify import notify

    result = await run_drift_scan(db, payload.ref_sessions, payload.cur_sessions)
    if result.get("breached"):
        await notify(db, kind="drift_breach", severity="warn",
                     title="input drift breached its threshold",
                     body=f"worst axis {result.get('worst_axis')} at {result.get('worst_score')}",
                     href="/govern", subject_type="drift", subject_id="input",
                     meta={k: result.get(k) for k in ("worst_axis", "worst_score", "threshold")})
    return result


@router.post("/govern/controller/tick")
async def controller_tick(schedule_bursts: bool = True, db: AsyncSession = Depends(db_session)):
    return await tick(db, schedule_bursts=schedule_bursts)


# ---- kill switch + state + audit ----
@router.get("/govern/state")
async def state(db: AsyncSession = Depends(db_session)):
    return await K.state_dict(db)


class EngageIn(BaseModel):
    reason: str = "manual kill switch"
    task: str = "detection"


@router.post("/govern/killswitch/engage")
async def killswitch_engage(payload: EngageIn, db: AsyncSession = Depends(db_session)):
    """Stop the loop, and say so loudly. Critical severity: this is the one event where a person finding
    out an hour later is itself the incident."""
    from services.notify import notify

    result = await K.engage(db, payload.reason, payload.task)
    await notify(db, kind="kill_switch", severity="critical", role="admin",
                 title=f"kill switch engaged on {payload.task}", body=payload.reason,
                 href="/govern", subject_type="killswitch", subject_id=payload.task)
    return result


@router.post("/govern/killswitch/release")
async def killswitch_release(db: AsyncSession = Depends(db_session)):
    from services.notify import notify

    result = await K.release(db)
    await notify(db, kind="kill_switch", severity="info", role="admin",
                 title="kill switch released", body="the loop is running again", href="/govern",
                 subject_type="killswitch", subject_id="released", supersede=False)
    return result


@router.get("/govern/audit")
async def audit(actor: str | None = None, limit: int = 100, db: AsyncSession = Depends(db_session)):
    return await list_audit(db, actor, limit)
