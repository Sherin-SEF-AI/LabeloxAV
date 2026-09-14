"""Off-hours hook: score the champion and its challengers over frames nobody has compared them on.

Every model this program builds is measured against a gold set that was frozen the day it was sealed, so
the only evidence a promotion decision has ever had is about 164 images chosen months ago. Meanwhile the
corpus takes in new sessions continuously and no model is ever compared on them.

A sweep takes the frames ingested since the last one, scores the champion and each recent challenger over
them, and files every disagreement. It never promotes anything and never writes a label. What it produces
is evidence and a worklist: the frames where two models read the same pixels differently are exactly the
frames a person's minute is worth most on.

The GPU discipline is the same one every other card-touching job here follows, and it matters more than
usual because two detectors are involved: they run one at a time inside `gpu_slot`, never both resident,
with `training_holds_gpu` checked between chunks so a training job that starts mid-sweep waits seconds
rather than the length of the sweep.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Frame, ModelRegistry
from db.models import Session as DbSession

log = get_logger("shadow_agent")

KIND = "shadow_sweep"
CREATED_BY = "scheduler"
# One night's budget. The sweep scores every frame twice (champion and challenger), so this is 4,000
# forward passes per challenger, which is a handful of GPU minutes at 960px on this card.
SHADOW_FRAMES_PER_NIGHT = 2000
# Frames per gpu_slot hold. Small enough that a training job waiting on the card gets it back quickly,
# large enough that the slot is not thrashed once per image.
SHADOW_CHUNK = 256
# A challenger is a model registered recently enough to still be a candidate. Older ones have either been
# promoted or passed over, and re-scoring them every night buys nothing.
CHALLENGER_WINDOW = timedelta(days=30)
MAX_CHALLENGERS = 2
# The operating point the comparison is made at when no fitted threshold exists for the champion. It is
# the ship default, and the sweep records which of the two it used so a reader is never left guessing.
DEFAULT_THRESHOLD = 0.5
MIN_FREE_VRAM_MB = 2500.0


async def _high_water_mark(db: AsyncSession) -> datetime | None:
    """The newest frame a previous **committed** sweep scored, or None when none has ever finished.

    Only committed runs count. A sweep that failed, refused or was reverted did not compare anything on
    those frames, and letting its mark stand would step the window past them permanently: the frames
    would be skipped by every later sweep and nothing would ever say so. Taking the maximum across all
    committed sweeps rather than the latest one is the same care in the other direction, so a sweep run
    out of order cannot pull the window backwards and re-score what has already been compared.
    """
    rows = (await db.execute(
        select(AgentRun.counts).where(AgentRun.kind == KIND, AgentRun.status == "committed")
        .order_by(AgentRun.created_at.desc()).limit(20))).scalars().all()
    marks = []
    for counts in rows:
        hwm = (counts or {}).get("high_water_mark")
        if not hwm:
            continue
        try:
            marks.append(datetime.fromisoformat(hwm))
        except ValueError:
            continue
    return max(marks) if marks else None


async def _frames_to_sweep(db: AsyncSession, hwm: datetime | None, limit: int) -> list[uuid.UUID]:
    """Real, selected, non-duplicate frames newer than the last sweep, oldest first.

    Oldest first so the high-water mark advances contiguously: taking the newest would leave a permanent
    hole in the middle that no later sweep ever revisits.
    """
    from core.origin import REAL

    stmt = (select(Frame.frame_id)
            .join(DbSession, Frame.session_id == DbSession.session_id)
            .where(Frame.origin == REAL, Frame.selected.is_(True),
                   DbSession.origin == REAL))
    # A duplicate frame scored twice is the same picture counted twice, which would weight whatever the
    # camera happened to sit still in front of.
    stmt = stmt.where((Frame.is_dup_canonical.is_(True)) | (Frame.dup_group_id.is_(None)))
    if hwm is not None:
        stmt = stmt.where(Frame.created_at > hwm)
    stmt = stmt.order_by(Frame.created_at.asc()).limit(limit)
    return list((await db.execute(stmt)).scalars().all())


async def _challengers(db: AsyncSession, champion: str) -> list[str]:
    since = datetime.now(UTC) - CHALLENGER_WINDOW
    rows = (await db.execute(
        select(ModelRegistry.model_version)
        .where(ModelRegistry.task == "detection", ModelRegistry.is_champion.is_(False),
               ModelRegistry.model_version != champion,
               ModelRegistry.weights_uri.isnot(None),
               ModelRegistry.created_at >= since)
        .order_by(ModelRegistry.created_at.desc()).limit(MAX_CHALLENGERS))).scalars().all()
    return list(rows)


async def _champion(db: AsyncSession) -> str | None:
    return (await db.execute(
        select(ModelRegistry.model_version).where(
            ModelRegistry.task == "detection", ModelRegistry.is_champion.is_(True))
        .limit(1))).scalar_one_or_none()


async def _score_in_chunks(db: AsyncSession, *, model_version: str, frame_ids: list[uuid.UUID],
                           sweep_id: str) -> tuple[str | None, dict]:
    """Score every frame with one model, one `gpu_slot` hold per chunk, yielding to training between them.

    Returns the inference run id and what the sweep had to wait for. Each chunk is its own `run_inference`
    call under its own scope, so a sweep that stops halfway has scored what it scored and can say so; the
    ids of every chunk run are returned for the matcher to read.
    """
    from core.gpu_slot import gpu_slot
    from services.labelops.class_precision import free_vram_mb
    from services.training.gpu_lease import training_holds_gpu
    from services.verdyx.inference_run import run_inference

    run_ids: list[str] = []
    stopped = None
    for i in range(0, len(frame_ids), SHADOW_CHUNK):
        if await training_holds_gpu(db):
            stopped = "training took the GPU"
            break
        free = await free_vram_mb()
        if free is not None and free < MIN_FREE_VRAM_MB:
            stopped = f"only {free:.0f} MB of VRAM free, need {MIN_FREE_VRAM_MB:.0f}"
            break
        chunk = frame_ids[i:i + SHADOW_CHUNK]
        async with gpu_slot(f"shadow:{model_version}", timeout_s=60):
            rid = await run_inference(db, model_version=model_version, frame_ids=chunk,
                                      scope={"shadow_sweep": sweep_id, "chunk": i // SHADOW_CHUNK})
        if rid is None:
            stopped = "the model has no downloadable weights"
            break
        run_ids.append(rid)
    return (run_ids[0] if run_ids else None), {"run_ids": run_ids, "stopped": stopped,
                                               "chunks": len(run_ids)}


async def maybe_shadow_sweep(db: AsyncSession, *, challengers: list[str] | None = None,
                             frame_limit: int | None = None) -> dict:
    """Score the champion and its challengers on frames nobody has compared them on.

    Called with no arguments this is the nightly hook: it finds its own challengers, runs once a day, and
    declines with a reason the rest of the time. Naming `challengers` makes it the operator's version of
    the same run, which is what the router exposes. Two guards behave differently between the two, and
    deliberately: the once-a-day marker exists to stop the scheduler running the same sweep twice, so an
    operator who asked for one is not refused by it, and the thirty-day discovery window exists to stop
    the scheduler re-scoring models that have already been passed over, so naming a model overrides it.
    Every guard that protects the machine rather than the schedule (the killswitch, the training lease,
    the VRAM floor) applies identically to both.
    """
    from services.agent.runtime.report import finish_run, launch, ran_since
    from services.govern.killswitch import get_state
    from services.training.gpu_lease import training_holds_gpu

    asked = list(challengers or [])
    st = await get_state(db)
    if not st.loop_enabled:
        return {"ran": False, "reason": "loop disabled by the killswitch"}
    if not asked and await ran_since(db, KIND, datetime.now(UTC) - timedelta(days=1)):
        return {"ran": False, "reason": "a sweep has already run today"}
    if await training_holds_gpu(db):
        return {"ran": False, "reason": "training holds the GPU"}

    champion = await _champion(db)
    if champion is None:
        return {"ran": False, "reason": "no detection champion is registered"}
    if asked:
        known = set((await db.execute(
            select(ModelRegistry.model_version).where(
                ModelRegistry.model_version.in_(asked),
                ModelRegistry.weights_uri.isnot(None)))).scalars().all())
        missing = [c for c in asked if c not in known]
        if missing:
            return {"ran": False,
                    "reason": f"not registered with downloadable weights: {', '.join(missing)}"}
        if champion in asked:
            return {"ran": False, "reason": "the champion cannot be its own challenger"}
        challengers = asked
    else:
        challengers = await _challengers(db, champion)
        if not challengers:
            return {"ran": False, "reason": "no challenger has been registered in the last 30 days"}

    hwm = await _high_water_mark(db)
    frames = await _frames_to_sweep(db, hwm, frame_limit or SHADOW_FRAMES_PER_NIGHT)
    if not frames:
        return {"ran": False, "reason": "no real selected frame has arrived since the last sweep"}

    async def worker(run_id):
        from db.session import get_sessionmaker
        from services.verdyx.shadow_run import (
            _threshold_for,
            match_predictions,
            queue_for_labelling,
        )

        status, report = "committed", {}
        try:
            async with get_sessionmaker()() as wdb:
                sweep_id = str(run_id)
                threshold = await _threshold_for(wdb, champion, DEFAULT_THRESHOLD)
                c_run, c_detail = await _score_in_chunks(
                    wdb, model_version=champion, frame_ids=frames, sweep_id=sweep_id)
                per_challenger = {}
                if c_run is None:
                    status = "refused"
                    report = {"reason": f"the champion scored nothing: {c_detail['stopped']}",
                              "champion": champion}
                else:
                    for ch in challengers:
                        x_run, x_detail = await _score_in_chunks(
                            wdb, model_version=ch, frame_ids=frames, sweep_id=sweep_id)
                        if x_run is None:
                            per_challenger[ch] = {"matched": False, "reason": x_detail["stopped"]}
                            continue
                        # Only the frames both models actually scored are comparable, and the chunk runs
                        # are paired in order because both models walked the same frame list.
                        matched = {"by_kind": {}, "written": 0, "frames": 0}
                        for cr, xr in zip(c_detail["run_ids"], x_detail["run_ids"], strict=False):
                            res = await match_predictions(wdb, sweep_run_id=run_id, champion_run=cr,
                                                          challenger_run=xr, threshold=threshold)
                            if "error" in res:
                                continue
                            matched["written"] += res["written"]
                            matched["frames"] += res["frames"]
                            for k, v in res["by_kind"].items():
                                matched["by_kind"][k] = matched["by_kind"].get(k, 0) + v
                        per_challenger[ch] = {"matched": True, **matched,
                                              "chunks": x_detail["chunks"], "stopped": x_detail["stopped"]}
                    # The worklist. Filed once for the whole sweep rather than per challenger, so a frame
                    # both challengers disagree with the champion on is opened once, not twice.
                    queued = await queue_for_labelling(wdb, sweep_run_id=run_id)
                    newest = (await wdb.execute(
                        select(func.max(Frame.created_at)).where(Frame.frame_id.in_(frames)))).scalar_one()
                    report = {"champion": champion, "challengers": challengers,
                              "frames_scored": len(frames), "threshold": threshold,
                              "threshold_source": ("fitted" if threshold != DEFAULT_THRESHOLD
                                                   else "ship default, none fitted"),
                              "champion_chunks": c_detail["chunks"],
                              "champion_stopped": c_detail["stopped"],
                              "per_challenger": per_challenger, "queued": queued,
                              "high_water_mark": newest.isoformat() if newest else None}
        except Exception as exc:  # noqa: BLE001 - a failed sweep records why, never leaves a run running
            status = "error"
            report = {"error": str(exc)[:400]}
            log.error("shadow.sweep_failed", error=str(exc))
        await finish_run(run_id, status=status, report=report)

    return {"ran": True, **(await launch(db, KIND, worker, created_by=CREATED_BY,
                                         policy={"champion": champion, "challengers": challengers,
                                                 "frames": len(frames)}))}


async def sweep_summary(db: AsyncSession, limit: int = 5) -> dict:
    """What the recent sweeps found and what still waits on a person, for `/autonomy`."""
    from db.models import ShadowDisagreement
    from services.verdyx.shadow_run import win_share

    runs = (await db.execute(select(AgentRun).where(AgentRun.kind == KIND)
                             .order_by(AgentRun.created_at.desc()).limit(limit))).scalars().all()
    by_state = {k: int(v) for k, v in (await db.execute(
        select(ShadowDisagreement.state, func.count()).group_by(ShadowDisagreement.state))).all()}
    by_kind = {k: int(v) for k, v in (await db.execute(
        select(ShadowDisagreement.kind, func.count()).group_by(ShadowDisagreement.kind))).all()}
    challenger_runs = (await db.execute(
        select(ShadowDisagreement.challenger_run_id).distinct())).scalars().all()
    shares = []
    for rid in challenger_runs:
        s = await win_share(db, rid)
        shares.append({"challenger_run_id": str(rid), **s})
    return {"sweeps": [{"run_id": str(r.run_id), "status": r.status,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                        "report": r.counts} for r in runs],
            "disagreements_by_state": by_state, "disagreements_by_kind": by_kind,
            "win_shares": shares}
