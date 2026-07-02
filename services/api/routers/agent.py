"""The annotation-agent API: dry-run a frame (plan), commit a reversible run, list/inspect runs, and revert
one. The plan endpoint writes nothing; commit auto-accepts the confident objects and routes the rest, all
recorded in one AgentRun; revert restores the exact prior state. Auto-accept is gated on reviewer role.
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AgentRun
from services.agent.flywheel import run_flywheel
from services.agent.frame_agent import commit_frame, plan_frame
from services.agent.policy import PolicyThresholds
from services.agent.reconcile import reconcile_frame
from services.agent.runs import list_runs, revert_run, run_dict
from services.api.deps import current_user, db_session, require_role

router = APIRouter()


class AgentPolicyIn(BaseModel):
    auto_accept_conf: float | None = None
    review_low: float | None = None
    require_agreement: bool | None = None


def _thresholds(body: AgentPolicyIn | None) -> PolicyThresholds:
    d = PolicyThresholds()
    if not body:
        return d
    return PolicyThresholds(
        auto_accept_conf=body.auto_accept_conf if body.auto_accept_conf is not None else d.auto_accept_conf,
        review_low=body.review_low if body.review_low is not None else d.review_low,
        require_agreement=body.require_agreement if body.require_agreement is not None else d.require_agreement,
    )


@router.post("/agent/frames/{frame_id}/plan", dependencies=[Depends(require_role("annotator"))])
async def plan(frame_id: str, body: AgentPolicyIn | None = None, db: AsyncSession = Depends(db_session)):
    """Dry-run: what the agent would auto-accept / route on this frame. Writes nothing."""
    try:
        return await plan_frame(db, uuid.UUID(frame_id), _thresholds(body))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/agent/frames/{frame_id}/run", dependencies=[Depends(require_role("reviewer"))])
async def run(frame_id: str, body: AgentPolicyIn | None = None,
              db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Commit the plan as one reversible run: auto-accept the confident, route the rest."""
    try:
        return await commit_frame(db, uuid.UUID(frame_id), _thresholds(body),
                                  created_by=str(user.user_id) if user else None)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/agent/coverage", dependencies=[Depends(require_role("annotator"))])
async def coverage(db: AsyncSession = Depends(db_session)):
    """Corpus coverage report: class balance, scene-axis coverage, geography, and the named gaps to fill."""
    from services.agent.coverage import analyze_coverage

    return await analyze_coverage(db)


class MineIn(BaseModel):
    session_id: str | None = None
    ttc_thresh: float = 2.5


@router.post("/agent/scenarios/mine", dependencies=[Depends(require_role("reviewer"))])
async def scenarios_mine(body: MineIn | None = None, db: AsyncSession = Depends(db_session)):
    """Mine safety-critical scenarios (near-miss / high-risk / hard-brake) into the scenario queue."""
    from services.agent.scenario_miner import mine_scenarios

    b = body or MineIn()
    return await mine_scenarios(db, b.session_id, ttc_thresh=b.ttc_thresh)


@router.post("/agent/disagreements/mine", dependencies=[Depends(require_role("reviewer"))])
async def disagreements_mine(body: MineIn | None = None, db: AsyncSession = Depends(db_session)):
    """Mine champion-vs-challenger model-disagreement frames (paths voted different classes) into the
    scenario queue -- the highest-value frames to label and an early regression signal."""
    from services.agent.disagreement import mine_disagreements

    b = body or MineIn()
    return await mine_disagreements(db, b.session_id)


@router.get("/agent/frames/{frame_id}/suggest", dependencies=[Depends(require_role("annotator"))])
async def suggest(frame_id: str, db: AsyncSession = Depends(db_session)):
    """Proactive assistant: the highest-leverage agent actions for this frame, each with its count."""
    from services.agent.copilot import suggest_for_frame

    return await suggest_for_frame(db, uuid.UUID(frame_id))


@router.get("/agent/report", dependencies=[Depends(require_role("annotator"))])
async def report(db: AsyncSession = Depends(db_session)):
    """Auto dataset report: corpus size, class balance, coverage gaps, fix-queue and scenario summaries."""
    from services.agent.copilot import dataset_report

    return await dataset_report(db)


class AskIn(BaseModel):
    text: str
    limit: int = 40


@router.post("/agent/ask", dependencies=[Depends(require_role("annotator"))])
async def ask(body: AskIn, db: AsyncSession = Depends(db_session)):
    """Conversational corpus query: ask the dataset a plain-language question ('pedestrians crossing against
    traffic at night') and get the matching frames, plus how the question was parsed into facets."""
    from services.agent.copilot import answer_corpus_query

    return await answer_corpus_query(db, body.text, limit=body.limit)


class CycleIn(BaseModel):
    max_frames: int = 25
    dry_run: bool = True
    retrain: bool = False


@router.post("/agent/training/cycle", dependencies=[Depends(require_role("reviewer"))])
async def training_cycle(body: CycleIn | None = None, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """One turn of the self-improving loop: flywheel tick (mine -> auto-accept/escalate) then, when not
    dry-run and retrain is set, a closed-loop retrain if enough corrections have accumulated."""
    from services.agent.training_daemon import flywheel_cycle

    b = body or CycleIn()
    return await flywheel_cycle(db, max_frames=b.max_frames, dry_run=b.dry_run, retrain=b.retrain,
                                created_by=str(user.user_id) if user else None)


@router.post("/agent/gold-drift", dependencies=[Depends(require_role("reviewer"))])
async def gold_drift(db: AsyncSession = Depends(db_session)):
    """Re-evaluate the serving champion on the gold set; roll back + pause the loop on regression."""
    from services.agent.training_daemon import check_gold_drift

    return await check_gold_drift(db)


class TemporalRepairIn(BaseModel):
    session_id: str | None = None
    min_majority: float = 0.8


@router.post("/agent/temporal-repair/plan", dependencies=[Depends(require_role("annotator"))])
async def temporal_repair_plan(body: TemporalRepairIn | None = None, db: AsyncSession = Depends(db_session)):
    """Dry-run: which class-flip outliers would be relabeled to their track majority. Writes nothing."""
    from services.agent.temporal_repair import plan_temporal_repair

    b = body or TemporalRepairIn()
    return await plan_temporal_repair(db, b.session_id, min_majority=b.min_majority)


@router.post("/agent/temporal-repair", dependencies=[Depends(require_role("reviewer"))])
async def temporal_repair_run(body: TemporalRepairIn | None = None,
                              db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Relabel strong-majority class-flip outliers to the track class as one reversible run."""
    from services.agent.temporal_repair import commit_temporal_repair

    b = body or TemporalRepairIn()
    return await commit_temporal_repair(db, b.session_id, min_majority=b.min_majority,
                                        created_by=str(user.user_id) if user else None)


class ErrorSweepIn(BaseModel):
    max_sessions: int = 10
    kinds: list[str] | None = None


@router.post("/agent/errors/sweep", dependencies=[Depends(require_role("reviewer"))])
async def error_sweep(body: ErrorSweepIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Launch the corpus-wide error daemon in the background: run every detector across sessions, refreshing
    the fix queue. Poll GET /agent/runs/{run_id} for progress, and GET /agent/errors/queue for the results."""
    from services.agent.error_daemon import run_error_sweep

    run_id = uuid.uuid4()
    run = AgentRun(run_id=run_id, kind="error_sweep", scope={"max_sessions": body.max_sessions},
                   status="running", policy={"kinds": body.kinds}, counts={}, changes={}, critic={},
                   created_by=str(user.user_id) if user else "daemon")
    db.add(run)
    await db.commit()
    asyncio.create_task(run_error_sweep(run_id, max_sessions=max(1, body.max_sessions), kinds=body.kinds))
    return {"run_id": str(run_id), "status": "running"}


@router.get("/agent/errors/queue", dependencies=[Depends(require_role("annotator"))])
async def error_queue(status: str = "pending", limit: int = 100, db: AsyncSession = Depends(db_session)):
    """The ranked fix queue: likely-wrong labels the daemon surfaced, worst first."""
    from services.errordetect.queue import list_candidates, summary

    return {"summary": await summary(db), "candidates": await list_candidates(db, status, limit)}


@router.post("/agent/frames/{frame_id}/attributes/plan", dependencies=[Depends(require_role("annotator"))])
async def attributes_plan(frame_id: str, db: AsyncSession = Depends(db_session)):
    """Dry-run: the derivable attributes (occlusion, truncation, static, direction) each machine object
    would gain. Writes nothing."""
    from services.agent.attribute_agent import plan_attributes

    try:
        return await plan_attributes(db, uuid.UUID(frame_id))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/agent/frames/{frame_id}/attributes", dependencies=[Depends(require_role("reviewer"))])
async def attributes_run(frame_id: str, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Fill the derivable attributes on the frame's objects as one reversible run."""
    from services.agent.attribute_agent import commit_attributes

    try:
        return await commit_attributes(db, uuid.UUID(frame_id), created_by=str(user.user_id) if user else None)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class CrossCamIn(BaseModel):
    tol_ms: int = 20
    high: float = 0.75
    min_vis: float = 0.50


@router.post("/agent/objects/{object_id}/crosscam/plan", dependencies=[Depends(require_role("annotator"))])
async def crosscam_plan(object_id: str, body: CrossCamIn | None = None, db: AsyncSession = Depends(db_session)):
    """Dry-run: which other cameras can see this object (via its 3D cuboid) and the box it would get. No writes."""
    from services.agent.crosscam_agent import plan_cross_camera

    b = body or CrossCamIn()
    try:
        return await plan_cross_camera(db, uuid.UUID(object_id), tol_ms=b.tol_ms, high=b.high, min_vis=b.min_vis)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/agent/objects/{object_id}/crosscam", dependencies=[Depends(require_role("reviewer"))])
async def crosscam_run(object_id: str, body: CrossCamIn | None = None,
                       db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Propagate the label to the other cameras that can see it, as one reversible run."""
    from services.agent.crosscam_agent import commit_cross_camera

    b = body or CrossCamIn()
    try:
        return await commit_cross_camera(db, uuid.UUID(object_id), tol_ms=b.tol_ms, high=b.high, min_vis=b.min_vis,
                                         created_by=str(user.user_id) if user else None)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class CuboidIn(BaseModel):
    min_iou: float = 0.35
    high: float = 0.60


@router.post("/agent/frames/{frame_id}/cuboids/plan", dependencies=[Depends(require_role("annotator"))])
async def cuboids_plan(frame_id: str, body: CuboidIn | None = None, db: AsyncSession = Depends(db_session)):
    """Dry-run: which 2D vehicle/VRU boxes on this frame lift to a valid 3D cuboid (monocular, reprojection-
    validated). Writes nothing."""
    from services.agent.cuboid_agent import plan_cuboids

    b = body or CuboidIn()
    try:
        return await plan_cuboids(db, uuid.UUID(frame_id), min_iou=b.min_iou, high=b.high)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/agent/frames/{frame_id}/cuboids", dependencies=[Depends(require_role("reviewer"))])
async def cuboids_run(frame_id: str, body: CuboidIn | None = None,
                      db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Attach fitted 3D cuboids to the frame's 2D objects as one reversible run."""
    from services.agent.cuboid_agent import commit_cuboids

    b = body or CuboidIn()
    try:
        return await commit_cuboids(db, uuid.UUID(frame_id), min_iou=b.min_iou, high=b.high,
                                    created_by=str(user.user_id) if user else None)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class PropagateIn(BaseModel):
    span: int = 24
    drift: float = 0.62
    high: float = 0.80


@router.post("/agent/objects/{object_id}/propagate/plan", dependencies=[Depends(require_role("annotator"))])
async def propagate_plan(object_id: str, body: PropagateIn | None = None, db: AsyncSession = Depends(db_session)):
    """Dry-run: what the track-propagation agent would carry across the clip from this keyframe (both
    directions, stopping where the box drifts). Writes nothing."""
    from services.agent.propagate_agent import plan_propagate

    b = body or PropagateIn()
    try:
        return await plan_propagate(db, uuid.UUID(object_id), span=b.span, drift=b.drift, high=b.high)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/agent/objects/{object_id}/propagate", dependencies=[Depends(require_role("reviewer"))])
async def propagate_run(object_id: str, body: PropagateIn | None = None,
                        db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Propagate the keyframe across its track and persist the boxes as one reversible run."""
    from services.agent.propagate_agent import commit_propagate

    b = body or PropagateIn()
    try:
        return await commit_propagate(db, uuid.UUID(object_id), span=b.span, drift=b.drift, high=b.high,
                                      created_by=str(user.user_id) if user else None)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class FlywheelIn(AgentPolicyIn):
    ticks: int = 1
    max_frames: int = 25
    session_id: str | None = None
    dry_run: bool = True  # default: report what it would auto-accept without writing


@router.post("/agent/flywheel", dependencies=[Depends(require_role("reviewer"))])
async def flywheel(body: FlywheelIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Launch the autonomous loop in the background: mine by value, auto-accept the sure ones / route the
    rest, then retrain if enough corrections have accumulated. Poll GET /agent/runs/{run_id} for progress.
    dry_run (default) plans only and writes nothing."""
    run_id = uuid.uuid4()
    run = AgentRun(
        run_id=run_id, kind="flywheel",
        scope={"ticks": body.ticks, "max_frames": body.max_frames, "session_id": body.session_id},
        status="running", policy=_thresholds(body).to_dict(), counts={}, changes={}, critic={},
        created_by=str(user.user_id) if user else "flywheel",
    )
    db.add(run)
    await db.commit()
    asyncio.create_task(run_flywheel(
        run_id, ticks=max(1, body.ticks), max_frames=max(1, body.max_frames),
        policy=_thresholds(body), session_id=body.session_id, dry_run=body.dry_run,
        created_by=str(user.user_id) if user else "flywheel",
    ))
    return {"run_id": str(run_id), "status": "running", "dry_run": body.dry_run}


class CommandIn(BaseModel):
    text: str
    frame_id: str


@router.post("/agent/command", dependencies=[Depends(require_role("annotator"))])
async def command(body: CommandIn, db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Natural-language control: turn an instruction ('auto-accept the two-wheelers above 0.9') into a
    scoped agent action on a frame. Returns the parsed intent, the result, and a plain-language summary.
    plan/find are read-only; accept/revert write and are reversible."""
    from services.agent.nl import execute_command
    from services.api.deps import role_rank

    can_write = bool(user) and role_rank(user.role) >= role_rank("reviewer")
    try:
        return await execute_command(db, body.text, uuid.UUID(body.frame_id),
                                     created_by=str(user.user_id) if user else None, can_write=can_write)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class FreshIn(AgentPolicyIn):
    commit: bool = False  # also apply the agent decision; False plans over the freshly-detected objects


@router.post("/agent/frames/{frame_id}/fresh", dependencies=[Depends(require_role("reviewer"))])
async def fresh(frame_id: str, body: FreshIn | None = None,
                db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Fresh-inference: detect the frame (Path A/B + fuse + gate + persist) then run the agent on the new
    objects in one shot. commit applies the decision; otherwise it plans. GPU-bound (reviewer role)."""
    from services.agent.fresh import label_and_decide

    body = body or FreshIn()
    try:
        return await label_and_decide(db, uuid.UUID(frame_id), commit=body.commit, policy=_thresholds(body),
                                      created_by=str(user.user_id) if user else None)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


class ReconcileIn(BaseModel):
    object_ids: list[str] | None = None  # None = the whole frame's machine objects
    apply: bool = False                  # apply strong 'correct' verdicts as reversible relabels
    apply_min_conf: float = 0.55


@router.post("/agent/frames/{frame_id}/reconcile", dependencies=[Depends(require_role("annotator"))])
async def reconcile(frame_id: str, body: ReconcileIn | None = None,
                    db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Adjudicate the frame's objects with an independent model (SigLIP 2 zero-shot): confirm / correct /
    unsure per object. Read-only unless apply=True, which relabels the strong 'correct' verdicts as one
    reversible AgentRun (reviewer role required to apply)."""
    from services.api.deps import role_rank

    body = body or ReconcileIn()
    if body.apply and not (user and role_rank(user.role) >= role_rank("reviewer")):
        raise HTTPException(403, "applying relabels requires reviewer role")
    try:
        return await reconcile_frame(db, uuid.UUID(frame_id), body.object_ids, apply=body.apply,
                                     apply_min_conf=body.apply_min_conf,
                                     created_by=str(user.user_id) if user else None)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class RelabelIn(BaseModel):
    min_conf: float = 0.45      # absolute floor the suggested class must clear
    margin: float = 0.15        # how far the suggested class must beat the current one


class RelabelAllIn(BaseModel):
    max_frames: int = 200
    session_id: str | None = None    # scope to one session, or None for the whole corpus
    min_conf: float = 0.45
    margin: float = 0.15


@router.post("/agent/frames/{frame_id}/relabel/plan", dependencies=[Depends(require_role("annotator"))])
async def relabel_plan(frame_id: str, body: RelabelIn | None = None, db: AsyncSession = Depends(db_session)):
    """Dry-run: which objects the reasoning layer would relabel and to what, with the winning class's
    confidence and whether it is decisive enough to keep or should route to review. Writes nothing."""
    from services.agent.relabel_agent import plan_relabel

    body = body or RelabelIn()
    try:
        return await plan_relabel(db, uuid.UUID(frame_id), min_conf=body.min_conf, margin=body.margin)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/agent/frames/{frame_id}/relabel", dependencies=[Depends(require_role("reviewer"))])
async def relabel_frame(frame_id: str, body: RelabelIn | None = None,
                        db: AsyncSession = Depends(db_session), user=Depends(current_user)):
    """Improve this frame's labels: an independent model re-reads every machine box and, where it decisively
    disagrees, corrects the class as one reversible run. Decisive corrections are kept; moderate ones are
    applied but routed to review for a human to confirm."""
    from services.agent.relabel_agent import commit_relabel

    body = body or RelabelIn()
    try:
        return await commit_relabel(db, uuid.UUID(frame_id), min_conf=body.min_conf, margin=body.margin,
                                    created_by=str(user.user_id) if user else None)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/agent/relabel/all", dependencies=[Depends(require_role("reviewer"))])
async def relabel_all(body: RelabelAllIn | None = None, db: AsyncSession = Depends(db_session),
                      user=Depends(current_user)):
    """Relabel all frames: launch the reasoning layer across the corpus (or one session) in the background,
    one reversible child run per frame. Poll GET /agent/runs/{run_id} for progress and totals."""
    from services.agent.relabel_agent import run_relabel_all

    body = body or RelabelAllIn()
    run_id = uuid.uuid4()
    run = AgentRun(run_id=run_id, kind="relabel_all", status="running",
                   scope={"max_frames": body.max_frames, "session_id": body.session_id},
                   policy={"min_conf": body.min_conf, "margin": body.margin}, counts={}, changes={}, critic={},
                   created_by=str(user.user_id) if user else "daemon")
    db.add(run)
    await db.commit()
    asyncio.create_task(run_relabel_all(run_id, max_frames=max(1, body.max_frames),
                                        session_id=body.session_id, min_conf=body.min_conf, margin=body.margin,
                                        created_by=str(user.user_id) if user else None))
    return {"run_id": str(run_id), "status": "running"}


class CleanupIn(BaseModel):
    do_pii: bool = True
    pii_limit: int = 5000


@router.post("/agent/cleanup-sweep", dependencies=[Depends(require_role("reviewer"))])
async def cleanup_sweep(body: CleanupIn | None = None, db: AsyncSession = Depends(db_session),
                        user=Depends(current_user)):
    """Apply the panoptic/quality gates to objects ALREADY in the corpus (no model re-run): remove boxed
    stuff (trees/barriers/sky), ego-hood boxes, oversize boxes, and duplicate/nested boxes, and backfill PII
    on pre-gate frames. Fast; fully reversible (removed objects are snapshotted). Poll GET /agent/runs/{id}."""
    from services.agent.cleanup_sweep import run_cleanup_sweep

    body = body or CleanupIn()
    run_id = uuid.uuid4()
    db.add(AgentRun(run_id=run_id, kind="cleanup_sweep", scope={}, status="running", policy=body.model_dump(),
                    counts={}, changes={}, critic={}, created_by=str(user.user_id) if user else "cleanup"))
    await db.commit()
    asyncio.create_task(run_cleanup_sweep(run_id, do_pii=body.do_pii, pii_limit=body.pii_limit))
    return {"run_id": str(run_id), "status": "running"}


class AuditIn(BaseModel):
    sample_size: int = 200
    vlm_calls: int = 60
    since_hours: int = 24


@router.post("/agent/audit/run", dependencies=[Depends(require_role("reviewer"))])
async def audit_run(body: AuditIn | None = None, db: AsyncSession = Depends(db_session),
                    user=Depends(current_user)):
    """Run the Overnight Auditor now: sample the window's auto-accepts, VLM + critic spot-check them within a
    token budget, fold in control-sample precision, and queue suspects to review as a reversible run. Poll
    GET /agent/audit/latest for the morning report."""
    from services.agent.overnight_auditor import launch_audit

    body = body or AuditIn()
    return await launch_audit(db, created_by=str(user.user_id) if user else "overnight_auditor",
                              sample_size=body.sample_size, vlm_calls=body.vlm_calls, since_hours=body.since_hours)


@router.get("/agent/audit/latest", dependencies=[Depends(require_role("annotator"))])
async def audit_latest(db: AsyncSession = Depends(db_session)):
    """The most recent Overnight Auditor run and its morning report."""
    from services.agent.overnight_auditor import latest_report

    return await latest_report(db) or {"report": None}


@router.get("/agent/runs", dependencies=[Depends(require_role("annotator"))])
async def runs(limit: int = 50, db: AsyncSession = Depends(db_session)):
    return await list_runs(db, limit)


@router.get("/agent/runs/{run_id}", dependencies=[Depends(require_role("annotator"))])
async def run_detail(run_id: str, db: AsyncSession = Depends(db_session)):
    r = await db.get(AgentRun, uuid.UUID(run_id))
    if r is None:
        raise HTTPException(404, "run not found")
    return {**run_dict(r), "changes": r.changes}


@router.post("/agent/runs/{run_id}/revert", dependencies=[Depends(require_role("reviewer"))])
async def revert(run_id: str, db: AsyncSession = Depends(db_session)):
    try:
        return await revert_run(db, uuid.UUID(run_id))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
