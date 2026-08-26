"""Cutting tracks where they stop being one object.

59.2% of tracks in this corpus contain a centre jump of more than a quarter of the frame width, and 45.9% of
track steps have no box overlap at all. One `track_id` inspected by hand spans 28 seconds, jumps from x=1887
to x=541, and is labelled truck, rider, autorickshaw, suv and motorcycle on consecutive frames of a single
receding vehicle. That is not a track, it is a bucket.

It matters beyond tidiness because everything downstream reads tracks as if they were one object: gap
filling interpolates between their endpoints, class corrections propagate along them, track events span
them, and exports ship them. A gap fill across such a track produced 137,913 objects at 0.209 precision.

The cause is in `services/autolabel/track/tracker.py`: the association gate is `iou >= iou_match OR
cos >= reid_cos`, so a DINOv3 cosine of 0.55 - which two arbitrary same-class vehicles routinely exceed -
creates a match at zero overlap. Fixing that changes tracking for new data. This repairs what is already
here, without re-tracking: a full `retrack_session` rewrites every `track_id`, and `TrackEvent`,
`GoldSet.track_ids` and the event proposals all reference them.

Splits are made on geometry alone (`services/temporal/gap_gate.py::is_discontinuity`), never on class. The
detector renames one object between frames constantly; splitting on that would shred correct tracks.

Reversible as one run per session. `services/temporal/reid.py::split_track` is the interactive path and
writes a `Review` row per moved object, which is right when a person splits a track and wrong for a sweep -
`review` means a human ruled, and three hundred thousand machine rows in it would corrupt every reader of
that table the same way a VLM verdict would.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Frame, Object, Track
from db.session import get_sessionmaker
from services.temporal.gap_gate import is_discontinuity

log = get_logger("quality.track_split")

KIND = "track_split"
MAX_CONSECUTIVE_FAILURES = 20

# A track is not cut into more pieces than this. A track needing more is not a track with a few bad joins,
# it is a bucket, and slicing it into forty fragments produces forty tracks of two objects that no consumer
# can use. Left whole and counted, so the tracker fix has something to be measured against.
MAX_CUTS_PER_TRACK = 12

_DETECTION_SOURCES = ("fused", "auto_accept", "imported", "human", "relabel", "vlm_review")


async def _cuts(db: AsyncSession, track_id) -> tuple[list[int], int]:
    """Timestamps where this track stops being one object, and how many members it has.

    Cuts are found on detector-sourced members only. An interpolated box sits by construction on the line
    between its anchors, so it never looks like a discontinuity and would mask the very join that produced
    it.
    """
    rows = (await db.execute(
        select(Object.bbox, Frame.ts_ns, Frame.width)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(Object.track_id == track_id, Object.source.in_(_DETECTION_SOURCES))
        .order_by(Frame.ts_ns))).all()
    if len(rows) < 3:
        return [], len(rows)
    cuts: list[int] = []
    for (b1, _t1, w), (b2, t2, _w2) in zip(rows, rows[1:], strict=False):
        if is_discontinuity(b1, b2, frame_width=float(w or 1920)):
            cuts.append(int(t2))
    return cuts, len(rows)


async def plan_track_split(db: AsyncSession, *, limit: int | None = None) -> dict:
    """What the sweep would do, computed exactly as the run computes it. Writes nothing."""
    tids = (await db.execute(select(Track.track_id).order_by(Track.track_id))).scalars().all()
    if limit:
        tids = tids[:limit]
    tracks_with_cuts = total_cuts = too_fragmented = 0
    worst: list[tuple[str, int]] = []
    for tid in tids:
        cuts, _n = await _cuts(db, tid)
        if not cuts:
            continue
        if len(cuts) > MAX_CUTS_PER_TRACK:
            too_fragmented += 1
            continue
        tracks_with_cuts += 1
        total_cuts += len(cuts)
        worst.append((str(tid), len(cuts)))
    worst.sort(key=lambda x: -x[1])
    return {"tracks_examined": len(tids), "tracks_with_cuts": tracks_with_cuts,
            "cuts": total_cuts, "new_tracks_expected": total_cuts,
            "too_fragmented_left_whole": too_fragmented,
            "worst": [{"track_id": t, "cuts": c} for t, c in worst[:10]]}


async def split_one(db: AsyncSession, track_id, cuts: list[int], run_id: uuid.UUID) -> int:
    """Cut one track at the given timestamps. Returns how many objects moved.

    Every moved object is stamped with the run id and its previous track recorded, which is what makes the
    whole sweep undoable. The new `Track` ids go in the run's policy so the revert can remove the rows it
    created rather than leaving empty tracks behind.
    """
    rows = (await db.execute(
        select(Object, Frame.ts_ns).join(Frame, Frame.frame_id == Object.frame_id)
        .where(Object.track_id == track_id).order_by(Frame.ts_ns))).all()
    if not rows:
        return 0
    track = await db.get(Track, track_id)
    if track is None:
        return 0

    moved = 0
    changes: dict[str, dict] = {}
    made: list[str] = []
    # Segment by cut point; everything before the first cut keeps the original track id, and each cut opens
    # a new one. Bounds are paired up front rather than looked up inside the loop: `bounds.index(lo)` is
    # quadratic and, worse, returns the first match, so two equal timestamps would silently segment wrong.
    ordered = sorted(set(cuts))
    spans = list(zip(ordered, [*ordered[1:], None], strict=True))
    for lo, hi in spans:
        seg = [(o, int(ts)) for o, ts in rows
               if int(ts) >= lo and (hi is None or int(ts) < hi)]
        if not seg:
            continue
        new_track = Track(session_id=track.session_id, class_id=track.class_id,
                          first_ts_ns=seg[0][1], last_ts_ns=seg[-1][1], trajectory=None,
                          tracker_version=track.tracker_version)
        db.add(new_track)
        await db.flush()
        made.append(str(new_track.track_id))
        for o, _ts in seg:
            changes[str(o.object_id)] = {"from_track": str(track_id)}
            o.provenance = {**(o.provenance or {}), "agent_run_id": str(run_id), "track_split": True}
            o.track_id = new_track.track_id
            moved += 1

    kept = [ts for _o, ts in rows if int(ts) < ordered[0]]
    if kept:
        track.last_ts_ns = max(int(t) for t in kept)
    # `trajectory` describes the whole of the old track and is now a statement about objects that left it.
    track.trajectory = None

    run = await db.get(AgentRun, run_id)
    if run is not None:
        run.changes = {**(run.changes or {}), **changes}
        pol = dict(run.policy or {})
        pol["created_tracks"] = [*pol.get("created_tracks", []), *made]
        run.policy = pol
    return moved


async def revert_split(db: AsyncSession, run: AgentRun) -> dict:
    """Put every moved object back on its original track and delete the tracks the run created.

    Its own revert rather than the generic one because the generic path restores class, state, source,
    cuboid and attrs and knows nothing about `track_id` - and because leaving the created `Track` rows
    behind would turn an undo into a corpus full of empty tracks.
    """
    restored = skipped = 0
    for oid, ch in (run.changes or {}).items():
        obj = await db.get(Object, uuid.UUID(oid))
        if obj is None or "from_track" not in ch:
            skipped += 1
            continue
        prov = obj.provenance or {}
        if str(prov.get("agent_run_id")) != str(run.run_id):
            skipped += 1        # something else owns it now
            continue
        obj.track_id = uuid.UUID(ch["from_track"])
        obj.provenance = {k: v for k, v in prov.items() if k not in ("agent_run_id", "track_split")}
        restored += 1

    await db.flush()
    removed = 0
    for tid in (run.policy or {}).get("created_tracks", []):
        t = await db.get(Track, uuid.UUID(tid))
        # Only if it is empty. A track that picked up objects after the split is somebody else's now.
        if t is None:
            continue
        n = (await db.execute(select(Object.object_id).where(Object.track_id == t.track_id).limit(1))).first()
        if n is None:
            await db.delete(t)
            removed += 1
    run.status = "reverted"
    await db.commit()
    out = {"run_id": str(run.run_id), "restored": restored, "skipped": skipped, "tracks_removed": removed}
    log.info("track_split.reverted", **out)
    return out


async def start_track_split(db: AsyncSession, *, created_by: str | None = None) -> dict:
    run = AgentRun(kind=KIND, status="running", scope={"what": "track_split"}, policy={}, counts={},
                   changes={}, critic={}, created_by=created_by)
    db.add(run)
    await db.commit()
    return {"run_id": str(run.run_id), "kind": KIND}


async def run_track_split(run_id: uuid.UUID, *, max_tracks: int = 20_000) -> None:
    """Cut every track that needs it, resumable, one session per transaction."""
    from services.agent.resume import beat, done_set

    maker = get_sessionmaker()
    async with maker() as db:
        prior = await db.get(AgentRun, run_id)
        done = done_set(dict(prior.progress or {})) if prior else set()
        totals = dict(prior.counts or {}) if prior else {}
        tids = [str(t) for t in
                (await db.execute(select(Track.track_id).order_by(Track.track_id))).scalars().all()]
    for k in ("tracks", "cut", "moved", "failed", "too_fragmented"):
        totals.setdefault(k, 0)

    targets = [t for t in tids if t not in done][:max_tracks]
    consecutive = 0
    error = None
    try:
        for tid in targets:
            try:
                async with maker() as db:
                    cuts, _n = await _cuts(db, uuid.UUID(tid))
                    if cuts and len(cuts) <= MAX_CUTS_PER_TRACK:
                        moved = await split_one(db, uuid.UUID(tid), cuts, run_id)
                        totals["cut"] += len(cuts)
                        totals["moved"] += moved
                        await db.commit()
                    elif cuts:
                        totals["too_fragmented"] += 1
                totals["tracks"] += 1
                consecutive = 0
            except Exception as exc:  # noqa: BLE001 - one bad track is not the corpus
                totals["failed"] += 1
                consecutive += 1
                log.warning("track_split.track_failed", track_id=tid, error=str(exc))
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"{MAX_CONSECUTIVE_FAILURES} tracks failed in a row; stopping") from exc
            done.add(tid)
            async with maker() as db:
                await beat(db, run_id, progress={"done": sorted(done), "total": len(tids)}, counts=totals)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    async with maker() as db:
        run = await db.get(AgentRun, run_id)
        if run is not None:
            # Committed even on abort: revert refuses anything not committed, and marking an aborted sweep
            # interrupted would strand every already-moved object with no way back.
            run.status, run.counts, run.error = "committed", totals, error
            await db.commit()
    log.info("track_split.done", run_id=str(run_id), **totals)
