"""Scheduled refresh of the measurements autonomy depends on, so none of them silently rots.

The measurement stack was nearly complete and nearly unused: per-class precision existed as a script
somebody ran twice, judge calibration was computed once (118 decisions, 2026-08) and never rebuilt as
human verdicts accrued, and detector judging had the same shape. Every one of them is a denominator
that settlement (plan phase 2) will make decisions against, and a denominator measured once is a
constant wearing a measurement's clothes.

Three hooks in the fleet idiom (self-guarding maybe_*, AgentRun spine, off-hours, GPU-aware):

- class precision: re-judge the classes whose newest verdict is older than the staleness bound,
  a few per night, so the sweep converges without ever owning the card for a whole night;
- judge calibration: rebuild when enough new human review decisions have accrued to move it -
  the corpus-pooled sens 0.76 / spec 0.80 was measured against 118 decisions, so +25 is material;
- detector judging: weekly, the ranker's machine weights come from it.

Staleness is the contract here: each refresh makes `measured_at` (the newest verdict's created_at)
current, and consumers are expected to refuse a measurement older than the bound rather than act on
it silently. The bounds live here, next to the machinery that keeps them satisfiable.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import MachineVerdict

log = get_logger("agent.measurement")

PRECISION_KIND = "class_precision_refresh"
CALIBRATION_KIND = "judge_calibration_refresh"
DETECTOR_KIND = "detector_judging"

# A precision measured this long ago describes a corpus that no longer exists (the flywheel relabels,
# the VLM lever promotes, humans rule). 14 days matches the cadence the remediation plan measured by.
PRECISION_STALE_DAYS = 14
# How many classes one night may re-judge. 3 classes x 120 crops on the local VLM is well under an
# off-hours window and leaves the card free for training and the embedder.
CLASSES_PER_NIGHT = 3
# New human review decisions since the last calibration that make a rebuild worth its VLM cost.
CALIBRATION_MIN_NEW = 25
DETECTOR_EVERY_DAYS = 7


async def stale_precision_classes(db: AsyncSession, *, bound_days: int = PRECISION_STALE_DAYS,
                                  limit: int = CLASSES_PER_NIGHT) -> list[dict]:
    """The measured-population classes whose precision is unmeasured or older than the bound.

    Ordered worst-first: never measured, then oldest measurement. The population threshold is
    class_targets' own (10,000 objects), so this refreshes the same set the script measured.
    """
    from services.labelops.class_precision import batch_id_for, class_targets

    targets = await class_targets(db)
    newest_by_batch = dict((await db.execute(
        select(MachineVerdict.batch_id, func.max(MachineVerdict.created_at))
        .where(MachineVerdict.batch_id.in_([batch_id_for(t["class_name"]) for t in targets]))
        .group_by(MachineVerdict.batch_id))).all())

    cutoff = datetime.now(UTC) - timedelta(days=bound_days)
    out = []
    for t in targets:
        newest = newest_by_batch.get(batch_id_for(t["class_name"]))
        if newest is None:
            out.append({**t, "measured_at": None, "reason": "never measured"})
        elif newest < cutoff:
            out.append({**t, "measured_at": newest.isoformat(),
                        "reason": f"measured {(datetime.now(UTC) - newest).days}d ago"})
    out.sort(key=lambda r: (r["measured_at"] is not None, r["measured_at"] or ""))
    return out[:limit]


async def maybe_refresh_class_precision(db: AsyncSession) -> dict:
    """Off-hours hook: re-judge up to CLASSES_PER_NIGHT stale classes. judge_class holds the GPU slot
    and re-checks headroom between batches itself, so a training job that starts mid-class waits
    seconds, not the length of the class."""
    from services.agent.runtime.report import finish_run, launch, ran_since
    from services.training.gpu_lease import training_holds_gpu

    day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    if await ran_since(db, PRECISION_KIND, day_start):
        return {"ran": False, "reason": "already ran today"}
    if await training_holds_gpu(db):
        return {"ran": False, "reason": "training holds the GPU"}

    stale = await stale_precision_classes(db)
    if not stale:
        return {"ran": False, "reason": "every measured class is within the staleness bound"}

    async def worker(run_id):
        from db.session import get_sessionmaker
        from services.labelops.class_precision import judge_class
        from services.labelops.vlm_review import judged_precision

        report: dict = {"classes": [], "bound_days": PRECISION_STALE_DAYS}
        status = "committed"
        try:
            for t in stale:
                async with get_sessionmaker()() as wdb:
                    res = await judge_class(wdb, t["class_name"])
                    summary = await judged_precision(wdb, f"class-precision:{t['class_name']}")
                report["classes"].append({
                    "class_name": t["class_name"], "reason": t["reason"],
                    "judged": res.get("judged"), "stalled": res.get("stalled"),
                    "raw": (summary.get("raw") or {}).get("p"),
                    "corrected": (summary.get("corrected") or {}).get("p")
                                 if summary.get("corrected") else None})
                if res.get("stalled"):
                    # The card went busy; whatever was judged is banked, the rest keeps its staleness
                    # and tomorrow's run picks it up. Stopping is the batch-by-batch contract.
                    report["stopped"] = "GPU headroom gone; remaining classes deferred to tomorrow"
                    break
        except Exception as exc:  # noqa: BLE001
            status = "error"
            report["error"] = str(exc)[:400]
            log.error("measurement.class_precision_failed", error=str(exc))
        await finish_run(run_id, status=status, report=report)

    return {"ran": True, "classes": [t["class_name"] for t in stale],
            **(await launch(db, PRECISION_KIND, worker, created_by="scheduler"))}


async def maybe_refresh_judge_calibration(db: AsyncSession) -> dict:
    """Off-hours hook: rebuild the judge's sens/spec once enough new human decisions have accrued.

    The trigger counts Review rows newer than the last calibration run, because human rulings are the
    only thing a recalibration can learn from; re-running on an unchanged evidence base would spend VLM
    calls to compute the same number.
    """
    from db.models import AgentRun, Review
    from services.agent.runtime.report import finish_run, latest_run, launch
    from services.training.gpu_lease import training_holds_gpu

    last = await latest_run(db, CALIBRATION_KIND)
    if last and last["status"] == "running":
        return {"ran": False, "reason": "a calibration is already running"}
    since = None
    if last:
        row = await db.get(AgentRun, uuid.UUID(last["run_id"]))
        since = row.created_at if row else None

    q = select(func.count()).select_from(Review)
    if since is not None:
        q = q.where(Review.created_at >= since)
    new_reviews = (await db.execute(q)).scalar_one()
    if new_reviews < CALIBRATION_MIN_NEW:
        return {"ran": False,
                "reason": f"only {new_reviews} new human decisions since the last calibration; "
                          f"the trigger is {CALIBRATION_MIN_NEW}"}
    if await training_holds_gpu(db):
        return {"ran": False, "reason": "training holds the GPU"}

    async def worker(run_id):
        from db.session import get_sessionmaker
        from services.labelops.judge_calibration import calibrate_judge

        status, report = "committed", {}
        try:
            async with get_sessionmaker()() as wdb:
                res = await calibrate_judge(wdb)
            report = {k: v for k, v in res.items() if k not in ("confusion",)}
            report["trigger_new_reviews"] = int(new_reviews)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            report = {"error": str(exc)[:400]}
            log.error("measurement.calibration_failed", error=str(exc))
        await finish_run(run_id, status=status, report=report)

    return {"ran": True, "new_reviews": int(new_reviews),
            **(await launch(db, CALIBRATION_KIND, worker, created_by="scheduler"))}


async def maybe_judge_detectors(db: AsyncSession) -> dict:
    """Off-hours hook: refresh every error detector's judged precision, weekly."""
    from services.agent.runtime.report import finish_run, launch, ran_since
    from services.training.gpu_lease import training_holds_gpu

    if await ran_since(db, DETECTOR_KIND, datetime.now(UTC) - timedelta(days=DETECTOR_EVERY_DAYS)):
        return {"ran": False, "reason": f"judged within the last {DETECTOR_EVERY_DAYS} days"}
    if await training_holds_gpu(db):
        return {"ran": False, "reason": "training holds the GPU"}

    async def worker(run_id):
        from db.session import get_sessionmaker
        from services.errordetect.judge_detectors import judge_all_detectors

        status, report = "committed", {}
        try:
            async with get_sessionmaker()() as wdb:
                res = await judge_all_detectors(wdb)
            report = {"per_kind": {k: {kk: vv for kk, vv in (v or {}).items() if kk != "objects"}
                                   for k, v in (res.get("per_kind") or {}).items()},
                      "sample_per_detector": res.get("sample_per_detector")}
        except Exception as exc:  # noqa: BLE001
            status = "error"
            report = {"error": str(exc)[:400]}
            log.error("measurement.detector_judging_failed", error=str(exc))
        await finish_run(run_id, status=status, report=report)

    return {"ran": True, **(await launch(db, DETECTOR_KIND, worker, created_by="scheduler"))}
