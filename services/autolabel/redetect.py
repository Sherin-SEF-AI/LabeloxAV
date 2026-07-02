"""Re-detection orchestration: clean the existing corpus with the new gates.

The thing/stuff filter, ego-hood mask, fusion de-duplication, and the oversize reviewer rule all live in the
detection path now, but they only shape NEW output. To fix frames already labelled, we re-run detection over
them. This orchestrates that safely on one GPU: first estimate each camera's ego mask (so the re-run can drop
the hood), then re-autolabel each session in turn (one at a time, yielding to training), and separately
backfill PII on any frame that predates the anonymization gate. Runs in the background; poll the per-session
autolabel jobs for progress.
"""

from __future__ import annotations

import uuid

from sqlalchemy import distinct, func, select

from core.logging import get_logger
from db.models import Frame
from db.models import Session as DbSession
from db.models import TrainingJob
from db.session import get_sessionmaker

log = get_logger("autolabel.redetect")


async def estimate_all_ego_masks(*, min_frames: int = 200, force: bool = False) -> dict:
    """Estimate and cache the hood mask for every camera with enough frames. Idempotent unless force."""
    from services.autolabel.ego_mask import clear_cache, estimate_ego_mask

    maker = get_sessionmaker()
    async with maker() as db:
        pairs = (await db.execute(
            select(DbSession.vehicle_id, Frame.cam_id, func.count())
            .join(Frame, Frame.session_id == DbSession.session_id)
            .group_by(DbSession.vehicle_id, Frame.cam_id)
            .having(func.count() >= min_frames))).all()

    out = {"cameras": 0, "with_hood": 0, "no_hood": []}
    for vehicle_id, cam_id, _n in pairs:
        out["cameras"] += 1
        mask = await estimate_ego_mask(vehicle_id, cam_id, force=force)
        if mask is not None:
            out["with_hood"] += 1
        else:
            out["no_hood"].append(f"{vehicle_id}/{cam_id}")
    clear_cache()   # so the runner's cached reads pick up the freshly-estimated masks
    log.info("redetect.ego_masks", **{k: v for k, v in out.items() if k != "no_hood"})
    return out


async def redetect_all_sessions(*, estimate_ego: bool = True, limit_per_session: int | None = None) -> dict:
    """Re-autolabel every session sequentially (single-GPU discipline). Estimates ego masks first so the
    re-run drops the hood. Skips while a training job holds the GPU. Returns per-session outcomes."""
    from services.autolabel.runner import autolabel_session

    maker = get_sessionmaker()
    async with maker() as db:
        if (await db.execute(select(TrainingJob.job_id).where(TrainingJob.status == "running").limit(1))).first():
            return {"skipped": "training job holds the GPU"}
        session_ids = list((await db.execute(select(distinct(Frame.session_id)))).scalars().all())

    ego = await estimate_all_ego_masks() if estimate_ego else None
    results = {"ego": ego, "sessions": 0, "objects": 0, "failed": []}
    for sid in session_ids:
        try:
            r = await autolabel_session(sid, limit_per_session)
            results["sessions"] += 1
            results["objects"] += int(r.get("objects", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            log.error("redetect.session_failed", session_id=str(sid), error=str(exc))
            results["failed"].append(str(sid))
    log.info("redetect.done", sessions=results["sessions"], objects=results["objects"], failed=len(results["failed"]))
    return results


async def redetect_and_backfill(run_id: uuid.UUID, *, estimate_ego: bool = True, backfill_pii: bool = True,
                                pii_limit: int = 5000) -> None:
    """Background entrypoint: PII-backfill the pre-gate frames (legal blocker), then re-detect every session
    with the new gates. Records the outcome on the AgentRun row."""
    from db.models import AgentRun

    maker = get_sessionmaker()
    outcome: dict = {}
    try:
        if backfill_pii:
            from services.anonymize.backfill import backfill_unaudited

            outcome["pii"] = await backfill_unaudited(limit=pii_limit)
        outcome["redetect"] = await redetect_all_sessions(estimate_ego=estimate_ego)
        status = "committed"
    except Exception as exc:  # noqa: BLE001
        log.error("redetect.failed", run_id=str(run_id), error=str(exc))
        outcome["error"] = str(exc)
        status = "error"
    async with maker() as db:
        run = await db.get(AgentRun, run_id)
        if run:
            run.status = status
            run.counts = outcome
            await db.commit()
