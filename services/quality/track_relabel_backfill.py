"""Applying every past class correction to the rest of the track it was made on.

A correction used to stop at the frame it was made on, because the frame editor never called the endpoint
that fixes a track. Measured before this sweep: 413 tracks carried an unambiguous human class, and of the
44,097 objects on them only 5,798 had it. The median track is 93 frames and the median number of frames a
person actually touched is 1, so 86.9% of every correction ever made was sitting on one frame while the
other 92 kept the detector's guess. Track `171228ff` is the shape of it: a human said `minivan` on one
frame and the other 118 read sedan, truck, container_truck, bus, cone and rider.

The editor now propagates as the correction is made, so this is about the backlog, not the mechanism.

SCOPE COMES FROM THE REVIEW LOG, NOT FROM `source`. The editor's ordinary geometry save sets
`source="human"` without touching the class, so selecting on the column over-selects every object anybody
ever dragged. A `Review` row whose `before.class_id` differs from its `after.class_id` is the precise
record of somebody deciding a class, which is the only thing worth fanning out.

A TRACK WHERE TWO PEOPLE CHOSE DIFFERENT CLASSES IS NEVER GUESSED AT. No majority, no most-recent, no
tiebreak: zero objects are touched and the competing decisions are reported. Two annotators disagreeing
about one track usually means the tracker stitched two objects together, and the fix for that is a merge or
a split, not a relabel that would make one of them permanently wrong.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Object, Review, Track
from services.agent.class_move import refuse_reason
from services.autolabel.ontology import get_ontology
from services.review_batch import KIND as BATCH_KIND
from services.review_batch import change_record

log = get_logger("track_relabel_backfill")

KIND = BATCH_KIND
# A handful of unreadable tracks is ordinary; twenty in a row is the database or the ontology being gone,
# and continuing would burn the backlog against a broken dependency.
MAX_CONSECUTIVE_FAILURES = 20


async def _human_class_by_track(db: AsyncSession) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Every track a person decided a class on, split into the unambiguous ones and the disputed ones."""
    rows = (await db.execute(
        select(Object.track_id, Review.before, Review.after)
        .join(Object, Object.object_id == Review.object_id)
        .where(Object.track_id.is_not(None)))).all()

    chosen: dict[str, set[int]] = {}
    for tid, before, after in rows:
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        b, a = before.get("class_id"), after.get("class_id")
        if a is None or b is None or int(a) == int(b):
            continue          # a geometry or state review, not a decision about the class
        chosen.setdefault(str(tid), set()).add(int(a))

    clear = {t: next(iter(c)) for t, c in chosen.items() if len(c) == 1}
    disputed = {t: sorted(c) for t, c in chosen.items() if len(c) > 1}
    return clear, disputed


def _known(onto) -> set[int]:
    """Class ids the loaded ontology can reason about.

    Three ids in the review log are not among them: custom classes added through the app after this
    ontology version was cut, including a typo pair (`back_side_of_autorikshasw` beside
    `back_side_of_autorikshaw`). A track whose human decision points at one of those is left alone and
    reported. Neither the class-move guard nor attribute validation can judge a class the ontology has
    never heard of, and fanning it across ninety frames on that basis is not a fix.
    """
    return {c.id for c in onto.classes}


async def plan_backfill(db: AsyncSession) -> dict:
    """What the sweep would do, without doing any of it.

    Run this first and reconcile it against the numbers in the module docstring. A sweep over 38,299 objects
    that turns out to have selected the wrong set is not something to discover afterwards.
    """
    onto = get_ontology()
    clear, disputed = await _human_class_by_track(db)
    known = _known(onto)
    unknown = {t: c for t, c in clear.items() if c not in known}
    clear = {t: c for t, c in clear.items() if c in known}

    # Per object, exactly as run_backfill decides, so the plan predicts the sweep rather than
    # approximating it. Judging the ontology guard once per track and dropping the whole track was the
    # first version of this and it under-reported by more than half, because a track can be partly
    # applicable: the guard is about the move from each object's current class, and a track carries
    # several.
    objects = stranded = refused = held = 0
    refused_detail: list[dict] = []
    for tid, cid in clear.items():
        rows = (await db.execute(select(Object.class_id, Object.source)
                                 .where(Object.track_id == uuid.UUID(tid)))).all()
        objects += len(rows)
        for class_id, source in rows:
            if int(class_id) == cid:
                continue
            if source == "human":
                held += 1
                continue
            reason = refuse_reason(onto, int(class_id), cid)
            if reason is not None:
                refused += 1
                if len(refused_detail) < 20:
                    refused_detail.append({"track_id": tid, "reason": reason,
                                           "from": _name(onto, int(class_id)), "to": _name(onto, cid)})
                continue
            stranded += 1

    return {
        "tracks_in_scope": len(clear), "objects_on_those_tracks": objects,
        "objects_the_sweep_would_write": stranded,
        "objects_held_by_a_human": held,
        "objects_refused_by_ontology": refused, "refused": refused_detail,
        "tracks_disputed": len(disputed),
        "disputed": [{"track_id": t, "classes": [_name(onto, c) for c in cs]}
                     for t, cs in list(disputed.items())[:20]],
        "tracks_unknown_class": len(unknown),
        "unknown_class_ids": sorted({c for c in unknown.values()}),
    }


def _name(onto, class_id: int) -> str:
    """A readable name, or the bare id when the ontology does not carry that class any more. This is a
    report, so an id it cannot name is information rather than a reason to fail."""
    try:
        return onto.by_id(int(class_id)).name
    except KeyError:
        return f"class {class_id}"


async def run_backfill(run_id: uuid.UUID, *, max_tracks: int = 5000) -> None:
    """Apply each track's human decision to the rest of its objects, resumably, as one revertible run."""
    from db.session import get_sessionmaker
    from services.agent.resume import beat, done_set

    maker = get_sessionmaker()
    onto = get_ontology()

    async with maker() as db:
        clear, disputed = await _human_class_by_track(db)
        prior = await db.get(AgentRun, run_id)
        done = done_set(dict(prior.progress or {})) if prior is not None else set()
        totals = dict(prior.counts or {}) if prior is not None else {}

    known = _known(onto)
    unknown = {t: c for t, c in clear.items() if c not in known}
    clear = {t: c for t, c in clear.items() if c in known}

    for k in ("tracks", "objects", "skipped_human", "refused"):
        totals.setdefault(k, 0)
    totals["disputed"] = len(disputed)
    totals["unknown_class"] = len(unknown)

    targets = [t for t in sorted(clear) if t not in done][:max_tracks]
    consecutive = 0
    try:
        for tid in targets:
            cid = clear[tid]
            try:
                async with maker() as db:
                    rows = (await db.execute(select(Object)
                                             .where(Object.track_id == uuid.UUID(tid),
                                                    Object.class_id != cid))).scalars().all()
                    changes: dict[str, dict] = {}
                    for o in rows:
                        if o.source == "human":
                            # Somebody ruled on this frame. The whole point of the sweep is to carry a human
                            # decision outward, not to overwrite a different one.
                            totals["skipped_human"] += 1
                            continue
                        reason = refuse_reason(onto, o.class_id, cid)
                        if reason is not None:
                            totals["refused"] += 1
                            continue
                        rec = change_record(o)   # captured before the mutation, or "before" is the new class
                        changes[str(o.object_id)] = rec
                        o.class_id = cid
                        o.source = "propagated"
                        o.state = "review"
                        o.version = (o.version or 1) + 1
                        o.provenance = {**(o.provenance or {}), "agent_run_id": str(run_id),
                                        "review_batch": True, "track_relabel_backfill": True}
                        db.add(Review(object_id=o.object_id, reviewer="track_relabel_backfill",
                                      user_id=None, action="reclassify_track_backfill",
                                      before={"class_id": rec["from_class"]},
                                      after={"class_id": cid}, time_spent_ms=0,
                                      ts_ns=int(datetime.now(UTC).timestamp() * 1_000_000_000)))

                    track = await db.get(Track, uuid.UUID(tid))
                    run = await db.get(AgentRun, run_id)
                    # Merged into the one run in the SAME transaction that stamps the objects, so there is
                    # never a window where rows are changed and unowned. Reassigned rather than mutated:
                    # these are JSONB and SQLAlchemy will not flag an in-place mutation as dirty, which
                    # would land tens of thousands of rows with no undo. `policy` needs a deep copy for
                    # the same reason and one level further down: dict(run.policy) copies the outer mapping
                    # while `track_class_from` stays the same object, so setdefault mutates in place, the
                    # reassigned value compares equal to the stored one and nothing is written. Measured
                    # the first time this ran: 1 of 300 prior track classes recorded.
                    if run is not None and changes:
                        run.changes = {**(run.changes or {}), **changes}
                        pol = json.loads(json.dumps(run.policy or {}))
                        pol.setdefault("track_class_from", {})[tid] = (
                            int(track.class_id) if track is not None else None)
                        pol.setdefault("track_class_to", {})[tid] = cid
                        run.policy = pol
                        # Only once something was actually written. A track whose every object the ontology
                        # guard refused keeps its own class: moving it alone would leave the track claiming
                        # a class none of its objects carry, which is the drift this sweep exists to
                        # reduce, and with nothing recorded it would not come back on revert.
                        if track is not None:
                            track.class_id = cid
                    await db.commit()

                totals["tracks"] += 1
                totals["objects"] += len(changes)
                consecutive = 0
            except Exception as exc:  # noqa: BLE001 - one bad track is not the backlog
                consecutive += 1
                log.warning("track_relabel_backfill.track_failed", track_id=tid, error=str(exc))
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"{consecutive} tracks in a row failed; stopping rather than walking the rest of "
                        "the backlog against a broken dependency") from exc

            done.add(tid)
            async with maker() as db:
                await beat(db, run_id, progress={"done": sorted(done), "total": len(clear)}, counts=totals)

        await _finish(maker, run_id, totals, None)
        log.info("track_relabel_backfill.done", run_id=str(run_id), **totals)
    except Exception as exc:  # noqa: BLE001
        # Committed, not interrupted, and deliberately so: everything already written carries this run's id
        # and is revertible, and services/agent/runs.py refuses to revert a run in any other status. Marking
        # it interrupted would strand thousands of changed rows with no way back, which is the one thing
        # this machinery exists to prevent.
        await _finish(maker, run_id, totals, f"interrupted: {exc}")
        log.error("track_relabel_backfill.aborted", run_id=str(run_id), error=str(exc), **totals)
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


async def start_backfill(db: AsyncSession, *, created_by: str | None = None) -> dict:
    """Create the run the sweep reports and reverts through. The caller spawns the work."""
    run = AgentRun(kind=KIND, status="running", scope={"what": "track_relabel_backfill"},
                   policy={}, counts={}, changes={}, critic={},
                   created_by=created_by or "track_relabel_backfill")
    db.add(run)
    await db.commit()
    return {"run_id": str(run.run_id), "kind": KIND}


__all__ = ["KIND", "MAX_CONSECUTIVE_FAILURES", "plan_backfill", "run_backfill", "start_backfill"]
