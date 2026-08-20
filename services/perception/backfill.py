"""Segmenting the drivable surface on every frame, locally, and recording what it could not do.

Drivable area existed on 1,643 of 41,752 frames when this was written, and the shape of that gap says why
it needed a runner rather than a button. `segment_drivable` had exactly one caller in the tree - the HTTP
route behind the editor's button - so a mask existed only where somebody had opened a frame and clicked,
and the main dashcam corpus (37,711 frames at 1920x1080) had seventy-nine of them.

The absence was not neutral. Lane proposals are filtered against the drivable mask
(services/autolabel/lane/plausible.py), and an absent mask is not an empty one: the filter returns
"plausible" when there is no mask to check against. So the lane plausibility gate was wired up and doing
nothing on 96% of frames.

AN UNSEGMENTABLE FRAME GETS A ROW. A frame the model finds no road in is written with empty polygons and
zero coverage, because "segmented, and there is no road here" and "never segmented" are different facts
and only recording the first lets coverage reach 100% honestly. A frame that could not be segmented at all
- unreadable image, model refused - is recorded in the run's counts and left WITHOUT a row, so it stays
visible as work still to do rather than being buried under a zero.

Shaped like services/agent/reanalyze.py, for the reasons that module records: a cursor of finished frame
ids so an interrupted pass resumes instead of restarting, a per-frame try/except so one unreadable image
cannot end a corpus sweep, a consecutive-failure breaker so a dead object store stops the run rather than
being burned through, and a heartbeat after every frame so the console shows progress rather than a
spinner.

Measured on an RTX 5080: mask2former-swin-large-mapillary is 1,532 MiB resident and 0.05 s per 1920x1080
frame, so the corpus is roughly half an hour of GPU. The model is loaded once and reused; the VRAM guard
in services/autolabel/drivable.py runs before that load and refuses rather than letting an OOM fall
through to the geometric placeholder.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import cv2
import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import AgentRun, DrivableMask, Frame

log = get_logger("drivable_backfill")

KIND = "drivable_backfill"
# Consecutive failures that end the run. A handful of unreadable frames is ordinary; twenty in a row is
# the object store or the GPU being gone, and continuing would burn the corpus against a broken dependency.
MAX_CONSECUTIVE_FAILURES = 20


async def coverage(db: AsyncSession) -> dict:
    """How much of the corpus carries a drivable mask, by capture size.

    Broken down because the headline hides the shape: the buckets that look finished are small older
    imports, and the main dashcam corpus is the one that is empty.
    """
    rows = (await db.execute(
        select(Frame.width, Frame.height, func.count(Frame.frame_id),
               func.count(DrivableMask.frame_id))
        .select_from(Frame)
        .outerjoin(DrivableMask, DrivableMask.frame_id == Frame.frame_id)
        .group_by(Frame.width, Frame.height)
        .order_by(func.count(Frame.frame_id).desc()))).all()
    total = sum(int(r[2]) for r in rows)
    covered = sum(int(r[3]) for r in rows)
    return {
        "frames": total, "covered": covered,
        "pct": round(100.0 * covered / total, 2) if total else 0.0,
        "by_size": [{"dims": f"{w}x{h}", "frames": int(n), "covered": int(c),
                     "pct": round(100.0 * int(c) / int(n), 1) if n else 0.0}
                    for w, h, n, c in rows],
    }


def _store_mask(store, frame: Frame, result: dict) -> str:
    """Same key and payload as the editor path, so one reader serves both.

    services/api/routers/drivable.py and services/perception/cloud.py already write this exact shape at
    this exact key; a third variant would mean a consumer that works on two thirds of the corpus.
    """
    key = f"masks/drivable/{frame.session_id}/{frame.frame_id}.json"
    payload = {"classes": result["classes"], "width": result["width"], "height": result["height"]}
    return store.put_bytes(key, json.dumps(payload).encode(), "application/json")


async def _pending(db: AsyncSession, *, session_id: str | None, limit: int,
                   redo: bool) -> list[Frame]:
    q = select(Frame).order_by(Frame.frame_id)
    if not redo:
        # Frames with no row at all. A row with zero coverage is a finished frame, not an unstarted one.
        q = q.where(~select(DrivableMask.frame_id).where(
            DrivableMask.frame_id == Frame.frame_id).exists())
    if session_id:
        q = q.where(Frame.session_id == uuid.UUID(session_id))
    return list((await db.execute(q.limit(limit))).scalars().all())


async def run_drivable_backfill(run_id: uuid.UUID, *, session_id: str | None = None,
                                max_frames: int = 50_000, redo: bool = False) -> None:
    """Segment every frame that has no mask, writing one row each, resuming from the run's cursor."""
    from db.session import get_sessionmaker
    from services.agent.resume import beat, done_set
    from services.autolabel.drivable import DrivableUnavailable, segment_drivable

    maker = get_sessionmaker()
    store = get_object_store()

    async with maker() as db:
        frames = await _pending(db, session_id=session_id, limit=max_frames, redo=redo)
        prior = await db.get(AgentRun, run_id)
        done = done_set(dict(prior.progress or {})) if prior is not None else set()
        totals = dict(prior.counts or {}) if prior is not None else {}

    for k in ("frames", "masks", "empty", "unreadable", "refused"):
        totals.setdefault(k, 0)

    if not frames:
        async with maker() as db:
            run = await db.get(AgentRun, run_id)
            if run:
                run.status = "committed"
                run.counts = {**totals, "detail": "every frame in scope already carries a drivable mask"}
                await db.commit()
        return
    if done:
        log.info("drivable_backfill.resuming", run_id=str(run_id), already_done=len(done))

    consecutive = 0
    try:
        for fr in frames:
            if str(fr.frame_id) in done:
                continue
            try:
                raw = store.get_bytes(fr.img_uri)
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("the stored image could not be decoded")
                # allow_trapezoid stays off: a geometric placeholder is indistinguishable from a real mask
                # to every consumer except model_version, and writing tens of thousands of them would be
                # worse than the gap this run exists to close.
                res = segment_drivable(img)
            except DrivableUnavailable as exc:
                # The model could not run. No row, so the frame stays visible as work rather than being
                # buried under a zero-coverage mask that reads as "checked, no road here".
                totals["refused"] += 1
                consecutive += 1
                log.warning("drivable_backfill.refused", frame=str(fr.frame_id), error=str(exc))
            except Exception as exc:  # noqa: BLE001 - one bad frame is not the corpus
                totals["unreadable"] += 1
                consecutive += 1
                log.warning("drivable_backfill.frame_failed", frame=str(fr.frame_id), error=str(exc))
            else:
                consecutive = 0
                uri = _store_mask(store, fr, res)
                async with maker() as db:
                    existing = await db.get(DrivableMask, fr.frame_id)
                    if existing is None:
                        db.add(DrivableMask(frame_id=fr.frame_id, mask_uri=uri,
                                            coverage=res["coverage"], source="proposed",
                                            model_version=res["model"]))
                    elif existing.source != "human":
                        # A human refinement is never overwritten by a machine pass.
                        existing.mask_uri = uri
                        existing.coverage = res["coverage"]
                        existing.model_version = res["model"]
                    await db.commit()
                totals["masks"] += 1
                if not any(res["coverage"].get(c, 0.0) > 0 for c in res["coverage"]):
                    # Recorded, not skipped: a frame with no road in it has been checked, and saying so is
                    # what lets coverage reach 100% without lying about what was found.
                    totals["empty"] += 1

            totals["frames"] += 1
            done.add(str(fr.frame_id))
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"{consecutive} frames in a row failed; stopping rather than walking the rest of the "
                    "corpus against a broken object store or an unavailable GPU")
            async with maker() as db:
                await beat(db, run_id, progress={"done": sorted(done), "total": len(frames)},
                           counts=totals)

        async with maker() as db:
            run = await db.get(AgentRun, run_id)
            if run:
                run.status = "committed"
                run.counts = totals
                run.finished_at = datetime.now(UTC)
                await db.commit()
        log.info("drivable_backfill.done", run_id=str(run_id), **totals)
    except Exception as exc:  # noqa: BLE001
        async with maker() as db:
            run = await db.get(AgentRun, run_id)
            if run:
                run.status = "interrupted"
                run.error = f"interrupted: {exc}"
                run.counts = totals
                await db.commit()
        log.error("drivable_backfill.aborted", run_id=str(run_id), error=str(exc), **totals)
        raise


async def start_backfill(db: AsyncSession, *, session_id: str | None = None,
                         max_frames: int = 50_000, redo: bool = False,
                         created_by: str | None = None) -> dict:
    """Create the AgentRun the sweep reports through, and return it. The caller spawns the work."""
    run = AgentRun(kind=KIND, status="running",
                   scope={"session_id": session_id, "max_frames": max_frames, "redo": redo},
                   policy={}, created_by=uuid.UUID(created_by) if created_by else None)
    db.add(run)
    await db.commit()
    return {"run_id": str(run.run_id), "kind": KIND,
            "scope": {"session_id": session_id, "max_frames": max_frames, "redo": redo}}


__all__ = ["run_drivable_backfill", "start_backfill", "coverage", "KIND", "MAX_CONSECUTIVE_FAILURES"]
