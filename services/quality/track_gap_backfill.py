"""Filling the frames where a tracked object blinks out and comes back.

An annotator steps to the next frame and an object that was there is gone, then it returns a frame or two
later. Measured: **9,460 of 11,287 tracks (84%) have gaps**, 137,960 frames in total, and the gaps are
short. On the worst track in the corpus, 392 missing frames arrive as 241 separate holes averaging 1.6
frames and never longer than 5. That is the tracker losing a box to an occlusion or a low score for a
moment, not the object leaving the scene.

`services/intelligence/propagate.py interpolate_track` was built for exactly this and had never run once:
it wrote `source="interp"`, which `ck_object_source` does not admit, so every call raised a check violation
and zero objects in the corpus carry it. With that fixed, one track fills in a second and the boxes land
between their bracketing keyframes.

A LINEAR BOX IS NOT A MEASUREMENT. These land at `state="annotate"` and `conf=0.5`, which is where
interpolate already put them: visible on the canvas so the track reads as continuous, and sitting in the
annotate queue rather than counting as a label somebody made.

Resumable and revertible, in the shape of services/perception/backfill.py and
services/quality/track_relabel_backfill.py: a cursor of finished track ids, a per-track try/except, a
consecutive-failure breaker, a heartbeat, and one AgentRun whose `changes` mark every created object so
services/agent/runs.py revert_run deletes them again.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Frame, Object, Track

log = get_logger("track_gap_backfill")

KIND = "track_gap_fill"
MAX_CONSECUTIVE_FAILURES = 20
# A gap longer than this is not a flicker. A linear box across fifty frames is a straight line drawn
# through whatever the object actually did, and on a turning vehicle it leaves the road. The measured
# distribution says almost nothing is near this: the worst track's longest hole is 5 frames.
MAX_GAP_FRAMES = 12


async def _gapped(db: AsyncSession) -> list[tuple[str, int]]:
    """Tracks whose own frames do not cover the span they occupy, worst first.

    The span is counted over the track's session rather than over the whole corpus, because a gap only
    means something relative to frames that exist to be filled.
    """
    t0, t1 = func.min(Frame.ts_ns).label("t0"), func.max(Frame.ts_ns).label("t1")
    spans = (select(Object.track_id.label("tid"), Track.session_id.label("sid"),
                    t0, t1, func.count().label("covered"))
             .join(Frame, Frame.frame_id == Object.frame_id)
             .join(Track, Track.track_id == Object.track_id)
             .where(Object.track_id.is_not(None))
             .group_by(Object.track_id, Track.session_id)).subquery()

    total = (select(func.count()).select_from(Frame)
             .where(Frame.session_id == spans.c.sid,
                    Frame.ts_ns >= spans.c.t0, Frame.ts_ns <= spans.c.t1)
             .scalar_subquery())

    rows = (await db.execute(
        select(spans.c.tid, (total - spans.c.covered).label("gap"))
        .where(total > spans.c.covered)
        .order_by((total - spans.c.covered).desc()))).all()
    return [(str(tid), int(gap)) for tid, gap in rows]


async def plan_gap_fill(db: AsyncSession) -> dict:
    """How many frames the sweep would create, without creating any of them."""
    gapped = await _gapped(db)
    return {
        "tracks_with_gaps": len(gapped),
        "frames_missing": sum(g for _, g in gapped),
        "worst": [{"track_id": t, "gap": g} for t, g in gapped[:10]],
    }


async def run_gap_fill(run_id: uuid.UUID, *, max_tracks: int = 20_000) -> None:
    """Fill every gapped track, resumably, as one revertible run."""
    from db.session import get_sessionmaker
    from services.agent.resume import beat, done_set
    from services.temporal.interpolate import interpolate_track_keyframed

    maker = get_sessionmaker()
    async with maker() as db:
        gapped = await _gapped(db)
        prior = await db.get(AgentRun, run_id)
        done = done_set(dict(prior.progress or {})) if prior is not None else set()
        totals = dict(prior.counts or {}) if prior is not None else {}

    for k in ("tracks", "created", "failed", "skipped_long_gaps"):
        totals.setdefault(k, 0)

    targets = [t for t, _ in gapped if t not in done][:max_tracks]
    consecutive = 0
    try:
        for tid in targets:
            try:
                res = await interpolate_track_keyframed(
                    uuid.UUID(tid), "cubic", anchor_policy="detection", run_id=run_id)
                totals["created"] += int(res.get("created") or 0)
                totals["refused_frames"] += int(res.get("refused_frames") or 0)
                for reason, n in (res.get("refused") or {}).items():
                    totals[f"refused_{reason}"] = totals.get(f"refused_{reason}", 0) + int(n)
                totals["tracks"] += 1
                consecutive = 0
            except Exception as exc:  # noqa: BLE001 - one bad track is not the corpus
                totals["failed"] += 1
                consecutive += 1
                log.warning("track_gap_backfill.track_failed", track_id=tid, error=str(exc))
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"{consecutive} tracks in a row failed; stopping rather than walking the rest of "
                        "the corpus against a broken dependency") from exc

            done.add(tid)
            async with maker() as db:
                await beat(db, run_id, progress={"done": sorted(done), "total": len(gapped)},
                           counts=totals)

        await _finish(maker, run_id, totals, None)
        log.info("track_gap_backfill.done", run_id=str(run_id), **totals)
    except Exception as exc:  # noqa: BLE001
        # Committed with the error recorded, not interrupted: everything already created carries this run's
        # id and is revertible, and services/agent/runs.py refuses to revert a run in any other status.
        await _finish(maker, run_id, totals, f"interrupted: {exc}")
        log.error("track_gap_backfill.aborted", run_id=str(run_id), error=str(exc), **totals)
        raise


async def _finish(maker, run_id: uuid.UUID, totals: dict, error: str | None) -> None:
    async with maker() as db:
        run = await db.get(AgentRun, run_id)
        if run is None:
            return
        run.status = "committed"
        run.counts = totals
        run.error = error
        await db.commit()


async def start_gap_fill(db: AsyncSession, *, created_by: str | None = None) -> dict:
    """Create the run the sweep reports and reverts through. The caller spawns the work."""
    run = AgentRun(kind=KIND, status="running", scope={"what": "track_gap_fill"},
                   policy={}, counts={}, changes={}, critic={},
                   created_by=created_by or "track_gap_fill")
    db.add(run)
    await db.commit()
    return {"run_id": str(run.run_id), "kind": KIND}


__all__ = ["KIND", "MAX_GAP_FRAMES", "plan_gap_fill", "run_gap_fill", "start_gap_fill"]
