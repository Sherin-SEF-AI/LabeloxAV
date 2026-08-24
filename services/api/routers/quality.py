"""Gate B (M9) endpoints: the quality sheet, gold-set listing, and the seal/fit triggers.

The sheet is served from cached metrics (no GPU in the request). Measuring is done out of band via
`make m9`. Sealing a gold set and fitting calibration are CPU/IO and run inline here.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.observability import spawn
from services.analytics.quality import list_gold_sets, quality_sheet
from services.api.deps import CalibrateFitIn, GoldSealIn, current_user, db_session, require_role
from services.autolabel.isotonic import fit_isotonic
from services.training.gold import GoldSpec, seal_gold

log = get_logger("quality_api")

router = APIRouter()


class IaaIn(BaseModel):
    set_a: list                     # [{bbox:[x1,y1,x2,y2], class_name}]
    set_b: list
    iou_thresh: float = 0.5


@router.post("/quality/iaa")
async def inter_annotator_agreement(body: IaaIn):
    """Milestone I: inter-annotator agreement between two independent label sets on a frame: detection
    agreement, class agreement, mean IoU, and Cohen's kappa."""
    from services.quality.iaa import iaa_score
    return iaa_score(body.set_a, body.set_b, body.iou_thresh)


@router.get("/quality/attr-audit/{session_id}")
async def attr_audit(session_id: str):
    """Milestone I: scan a session's objects for attribute-schema violations against each object's current
    class (surfaces labels invalidated by a class change or written before validation)."""
    from uuid import UUID

    from services.quality.attr_audit import session_attr_audit
    return await session_attr_audit(UUID(session_id))


@router.get("/quality/gold-sets")
async def gold_sets():
    return await list_gold_sets()


@router.get("/quality/sheet")
async def sheet(gold_id: str):
    res = await quality_sheet(gold_id)
    if not res.get("found"):
        raise HTTPException(status_code=404, detail=f"gold set {gold_id} not found")
    return res


@router.post("/quality/gold/{gold_id}/measure", dependencies=[Depends(require_role("reviewer"))])
async def measure(gold_id: str, db: AsyncSession = Depends(db_session)):
    """Score a sealed gold set, in the background.

    The page told the operator to run `make m9 ARGS="--gold <id>"` from a shell, which is not something a
    reviewer has, so a sealed set stayed unmeasured until an engineer happened to do it. GPU work, so it
    yields to a training job exactly as the other GPU routes do, and it refuses a set whose objects no
    longer exist rather than spending the GPU to measure nothing: five of the seven on this deployment are
    in that state.
    """
    from services.analytics.quality import list_gold_sets, measure_gold
    from services.training.gpu_lease import training_holds_gpu

    if await training_holds_gpu(db):
        raise HTTPException(503, "GPU reserved for a training job; measurement is paused until it finishes")

    row = next((g for g in await list_gold_sets() if g["gold_id"] == gold_id), None)
    if row is None:
        raise HTTPException(404, f"gold set {gold_id} not found")
    if not row["usable"]:
        raise HTTPException(409,
                            f"{row['n_missing']} of {row['n_objects']} objects in this gold set no longer "
                            "exist, so there is nothing left to measure against")

    async def _run() -> None:
        try:
            await measure_gold(gold_id)
        except Exception as exc:  # noqa: BLE001
            log.error("quality.measure_failed", gold_id=gold_id, error=str(exc))

    spawn(_run(), name="_measure_gold")
    return {"started": True, "gold_id": gold_id, "n_alive": row["n_alive"]}


# Sealing the gold set fixes the ground truth every evaluation, every promotion gate and every export
# certificate is scored against. Its sibling /quality/gold/{id}/measure was already reviewer-gated;
# this was not, so the annotator floor applied and anyone signed in could reseal it.
@router.post("/quality/gold/seal", dependencies=[Depends(require_role("reviewer"))])
async def seal(payload: GoldSealIn):
    spec = GoldSpec(name=payload.name, cities=payload.cities, session_id=payload.session_id,
                    class_names=payload.class_names, limit=payload.limit)
    try:
        return await seal_gold(spec)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# Fitting the isotonic curve changes what every confidence in the system means, and the auto-accept
# threshold is read against it.
@router.post("/quality/calibrate/fit", dependencies=[Depends(require_role("reviewer"))])
async def calibrate_fit(payload: CalibrateFitIn):
    try:
        return await fit_isotonic(payload.gold_id, payload.session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# Carrying every past class correction across the rest of the track it was made on.
#
# Admin rather than reviewer: this is a corpus-wide rewrite of tens of thousands of rows, not a decision
# about one thing. `plan` writes nothing and exists to be reconciled against the measured numbers before
# `commit` is allowed anywhere near the corpus.
@router.get("/quality/track-relabel-backfill/plan", dependencies=[Depends(require_role("admin"))])
async def track_relabel_plan(db: AsyncSession = Depends(db_session)):
    from services.quality.track_relabel_backfill import plan_backfill

    return await plan_backfill(db)


@router.post("/quality/track-relabel-backfill", dependencies=[Depends(require_role("admin"))])
async def track_relabel_backfill(max_tracks: int = 5000, db: AsyncSession = Depends(db_session),
                                 user=Depends(current_user)):
    from core.observability import spawn
    from services.quality.track_relabel_backfill import run_backfill, start_backfill

    res = await start_backfill(db, created_by=str(user.user_id) if user else None)
    spawn(run_backfill(UUID(res["run_id"]), max_tracks=max_tracks), name="track_relabel_backfill")
    return res


# Filling the frames where a tracked object blinks out and comes back. 9,460 of 11,287 tracks have gaps,
# 137,960 frames, in holes that average under two frames. Admin, because it creates objects corpus-wide.
@router.get("/quality/track-gap-fill/plan", dependencies=[Depends(require_role("admin"))])
async def track_gap_plan(db: AsyncSession = Depends(db_session)):
    from services.quality.track_gap_backfill import plan_gap_fill

    return await plan_gap_fill(db)


@router.post("/quality/track-gap-fill", dependencies=[Depends(require_role("admin"))])
async def track_gap_fill(max_tracks: int = 20_000, db: AsyncSession = Depends(db_session),
                         user=Depends(current_user)):
    from core.observability import spawn
    from services.quality.track_gap_backfill import run_gap_fill, start_gap_fill

    res = await start_gap_fill(db, created_by=str(user.user_id) if user else None)
    spawn(run_gap_fill(UUID(res["run_id"]), max_tracks=max_tracks), name="track_gap_fill")
    return res
