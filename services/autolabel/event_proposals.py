"""Two heuristic track-event proposers, and only two.

Both read `ObjectDynamics`, which is 567,475 rows of per-object motion state. **Those rows are IPM monocular
estimates.** There is no LiDAR behind them and no ego heading anywhere in the schema: `frame.ego_speed` is
populated on 6 frames of 41,752. So `speed_kmh` is a projection through an assumed ground plane, and a
pothole that pitches the camera moves every speed in the frame. Everything here lands `state="proposed"` with
its evidence attached, and a person decides.

The other twenty-one event types in the AV vocabulary are human-labeled, and `TrackEventType.proposable`
says which two are not. That flag exists so this file is not "completed" later by writing twenty-one more
heuristics for manoeuvres a monocular estimate cannot see. `blocking_intersection` needs to know where the
junction is; `overtaking_on_left` needs to know which lane is which; `looking_at_vehicle` needs a head pose.
None of those signals exist here, and a heuristic that guesses at them produces proposals whose rejection
rate is the only thing it measures.

Idempotent: a track that already carries an event of the same type overlapping the proposed span is left
alone, so re-running over a session does not stack duplicates on the tracks that fire every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import Frame, ObjectDynamics, Track, TrackEvent
from services.autolabel.ontology import get_ontology
from services.domain import validate_track_event_type

log = get_logger("autolabel.event_proposals")

_NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class Sample:
    """One dynamics row reduced to what a proposer reads."""

    frame_id: UUID
    ts_ns: int
    speed_kmh: float
    lateral_m: float | None


@dataclass(frozen=True)
class Span:
    """A proposal before it becomes a row."""

    event_type: str
    start_frame_id: UUID
    end_frame_id: UUID
    start_ts_ns: int
    end_ts_ns: int
    confidence: float
    evidence: dict


def _usable(rows: list[ObjectDynamics], cfg) -> list[Sample]:
    """Ordered samples with a speed, dropping rows the estimator itself was unsure about."""
    out = [
        Sample(r.frame_id, r.ts_ns, float(r.speed_kmh),
               None if r.lateral_m is None else float(r.lateral_m))
        for r in rows
        if r.frame_id is not None and r.ts_ns is not None and r.speed_kmh is not None
        and (r.confidence or 0.0) >= cfg.min_dynamics_confidence
        # Not a measurement. See TrackEventSettings.max_plausible_kmh for what the column actually holds.
        and float(r.speed_kmh) <= cfg.max_plausible_kmh
    ]
    return sorted(out, key=lambda s: s.ts_ns)


def _stays_down(samples: list[Sample], end_i: int, landed_kmh: float, cfg) -> bool:
    """True when the speed remains near where it landed for `brake_hold_s` after the drop.

    A vehicle that brakes hard is slow for a moment afterwards. An estimator that glitched for one frame is
    back where it started on the next, and this is the check that tells those apart. A track that simply
    ends at the drop counts as held: the evidence for a rebound is absent, not negative.
    """
    hold_ns = int(cfg.brake_hold_s * _NS_PER_S)
    end_ts = samples[end_i].ts_ns
    for s in samples[end_i + 1:]:
        if s.ts_ns - end_ts > hold_ns:
            return True
        if s.speed_kmh > landed_kmh + cfg.brake_hold_tol_kmh:
            return False
    return True


def propose_hard_brake(samples: list[Sample], cfg) -> list[Span]:
    """Spans where speed falls by `brake_drop_kmh` inside `brake_window_s`.

    The span runs from the last sample before the fall began to the sample where the speed stops falling,
    which is what an annotator would draw: the event is the deceleration, not the low speed afterwards.

    Overlapping candidates are merged rather than emitted separately. A long deceleration satisfies the
    window at many starting points, and one brake is one event.
    """
    if len(samples) < cfg.min_samples:
        return []
    window_ns = int(cfg.brake_window_s * _NS_PER_S)
    spans: list[Span] = []
    i = 0
    while i < len(samples) - 1:
        a = samples[i]
        best = None
        for j in range(i + 1, len(samples)):
            b = samples[j]
            if b.ts_ns - a.ts_ns > window_ns:
                break
            drop = a.speed_kmh - b.speed_kmh
            if drop >= cfg.brake_drop_kmh and (best is None or drop > best[1]):
                best = (b, drop, j)
        if best is None:
            i += 1
            continue
        b, drop, j = best
        # Extend while the speed keeps falling: the brake has not ended at the window edge.
        k = j
        while k + 1 < len(samples) and samples[k + 1].speed_kmh < samples[k].speed_kmh:
            k += 1
        end = samples[k]

        # Shape, not size. The measured noise floor on this signal is a median 9.1 km/h between consecutive
        # samples, so a drop the size of a real brake is about one sigma and a threshold alone detects the
        # estimator. What noise does not do is fall monotonically across four samples and then stay down.
        span = samples[i:k + 1]
        if len(span) < cfg.brake_min_samples:
            i += 1
            continue
        if any(span[n + 1].speed_kmh > span[n].speed_kmh for n in range(len(span) - 1)):
            i += 1
            continue
        if not _stays_down(samples, k, end.speed_kmh, cfg):
            i += 1
            continue

        dt_s = max((end.ts_ns - a.ts_ns) / _NS_PER_S, 1e-6)
        spans.append(Span(
            "hard_brake", a.frame_id, end.frame_id, a.ts_ns, end.ts_ns,
            # Confidence in the shape, not in the estimate. It rises with how far past the threshold the
            # drop went and is capped well below certainty because the speed itself is a projection.
            confidence=round(min(0.85, 0.4 + 0.4 * (drop / max(cfg.brake_drop_kmh, 1e-6) - 1.0)), 3),
            evidence={"speed_from_kmh": round(a.speed_kmh, 1), "speed_to_kmh": round(end.speed_kmh, 1),
                      "drop_kmh": round(a.speed_kmh - end.speed_kmh, 1), "over_s": round(dt_s, 2),
                      "n_samples": len(span), "held_below_kmh": round(end.speed_kmh + cfg.brake_hold_tol_kmh, 1),
                      "method": "ipm_mono_dynamics"},
        ))
        i = k + 1  # one brake is one event
    return spans


def propose_stopping_in_live_lane(samples: list[Sample], cfg) -> list[Span]:
    """Spans of at least `stop_min_s` at or under `stop_speed_kmh`, within `stop_lateral_m` of the ego path.

    The lateral bound is the whole discriminator. A vehicle stopped 8m to the side is parked at the kerb and
    is not a planning problem; the same vehicle stopped 1m off the ego path is the bus-at-a-stop case this
    event exists for. A sample with no lateral estimate ends the run rather than extending it: an unbounded
    stop is exactly the parked case, and guessing it into the live lane is the wrong direction to be wrong in.
    """
    if len(samples) < cfg.min_samples:
        return []
    min_ns = int(cfg.stop_min_s * _NS_PER_S)
    spans: list[Span] = []
    run: list[Sample] = []

    def flush() -> None:
        if len(run) < 2 or run[-1].ts_ns - run[0].ts_ns < min_ns:
            return
        laterals = [abs(s.lateral_m) for s in run if s.lateral_m is not None]
        spans.append(Span(
            "stopping_in_live_lane", run[0].frame_id, run[-1].frame_id, run[0].ts_ns, run[-1].ts_ns,
            confidence=round(min(0.8, 0.35 + 0.15 * ((run[-1].ts_ns - run[0].ts_ns) / min_ns)), 3),
            evidence={"held_s": round((run[-1].ts_ns - run[0].ts_ns) / _NS_PER_S, 2),
                      "max_speed_kmh": round(max(s.speed_kmh for s in run), 1),
                      "max_lateral_m": round(max(laterals), 2) if laterals else None,
                      "n_samples": len(run), "method": "ipm_mono_dynamics"},
        ))

    for s in samples:
        in_lane = (s.speed_kmh <= cfg.stop_speed_kmh and s.lateral_m is not None
                   and abs(s.lateral_m) <= cfg.stop_lateral_m)
        if in_lane:
            run.append(s)
        else:
            flush()
            run = []
    flush()
    return spans


async def _existing(db: AsyncSession, track_id: UUID) -> list[TrackEvent]:
    return list((await db.execute(
        select(TrackEvent).where(TrackEvent.track_id == track_id))).scalars())


def _already_covered(spans: list[TrackEvent], s: Span) -> bool:
    """True when an event of this type already overlaps this span in time."""
    return any(e.event_type == s.event_type and e.start_ts_ns <= s.end_ts_ns and e.end_ts_ns >= s.start_ts_ns
               for e in spans)


async def propose_for_track(db: AsyncSession, track_id: UUID, *, pack_id: str | None = None,
                            commit: bool = True) -> dict:
    """Run both proposers over one track and write what applies. Returns a summary, never raises on a track
    with no dynamics."""
    cfg = get_settings().track_events
    track = await db.get(Track, track_id)
    if track is None:
        return {"track_id": str(track_id), "proposed": [], "skipped": "track not found"}

    rows = list((await db.execute(
        select(ObjectDynamics).where(ObjectDynamics.track_id == track_id))).scalars())
    samples = _usable(rows, cfg)
    if len(samples) < cfg.min_samples:
        return {"track_id": str(track_id), "proposed": [],
                "skipped": f"{len(samples)} usable dynamics rows, need {cfg.min_samples}"}

    onto = get_ontology()
    try:
        l1 = onto.by_id(track.class_id).l1
    except KeyError:
        l1 = None

    candidates = propose_hard_brake(samples, cfg) + propose_stopping_in_live_lane(samples, cfg)
    existing = await _existing(db, track_id)
    written: list[str] = []
    for c in candidates:
        # The same applicability rule the router uses. A hard_brake on a pedestrian track is not a proposal
        # worth a reviewer's time, and the pack is what says so.
        if validate_track_event_type(c.event_type, l1, pack_id):
            continue
        if _already_covered(existing, c):
            continue
        db.add(TrackEvent(
            track_id=track_id, event_type=c.event_type,
            start_frame_id=c.start_frame_id, end_frame_id=c.end_frame_id,
            start_ts_ns=c.start_ts_ns, end_ts_ns=c.end_ts_ns,
            source="heuristic", state="proposed", confidence=c.confidence, evidence=c.evidence,
            created_by="event_proposals", ontology_version=onto.version))
        written.append(c.event_type)
    if commit and written:
        await db.commit()
    return {"track_id": str(track_id), "proposed": written, "n_samples": len(samples)}


async def propose_for_session(db: AsyncSession, session_id: UUID, *, limit: int | None = None,
                              pack_id: str | None = None) -> dict:
    """Every track in a session, one commit at the end.

    Batched deliberately: a session runs to a few thousand tracks and a commit per track is a few thousand
    round trips for a pass that is meant to run off-hours without competing with annotators.
    """
    q = select(Track.track_id).where(Track.session_id == session_id).order_by(Track.first_ts_ns)
    if limit:
        q = q.limit(limit)
    ids = list((await db.execute(q)).scalars())
    counts: dict[str, int] = {}
    for tid in ids:
        res = await propose_for_track(db, tid, pack_id=pack_id, commit=False)
        for name in res["proposed"]:
            counts[name] = counts.get(name, 0) + 1
    await db.commit()
    log.info("event_proposals.session", session_id=str(session_id), tracks=len(ids), proposed=counts)
    return {"session_id": str(session_id), "tracks": len(ids), "proposed": counts}


async def frames_for_span(db: AsyncSession, event: TrackEvent) -> list[Frame]:
    """The frames a span covers, for a reviewer stepping through it."""
    track = await db.get(Track, event.track_id)
    return list((await db.execute(
        select(Frame).where(Frame.session_id == track.session_id,
                            Frame.ts_ns >= event.start_ts_ns, Frame.ts_ns <= event.end_ts_ns)
        .order_by(Frame.ts_ns))).scalars())
