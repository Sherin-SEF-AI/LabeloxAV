"""Was the ego still closing on the signal while it was red?

The question worth answering is whether the vehicle entered on red, and the honest answer is that this
corpus cannot support it. Ego speed exists on 846 frames of 40,222, and of the 173 sessions carrying signal
phases, exactly none have it. A feature built on `ego_speed` would return nothing on every session anybody
opened, which is the failure this module exists to avoid rather than repeat.

What the corpus does have is 3,061 traffic-signal tracks spanning three or more frames, and a signal's
apparent geometry is an ego-motion instrument. Approaching a fixed object makes it grow and drift down the
frame; passing under one makes it leave through the top. So the claim here is deliberately the weaker,
supportable one: the ego was *closing on* a signal that was red, and kept closing.

Stated as approach rather than entry because that is what the evidence carries. A box that grows says the
gap is shrinking; it does not say the vehicle crossed the stop line, and no amount of monocular geometry
will say so without calibration. Naming it `signal_approach_on_red` rather than `ran_red_light` is the
difference between a finding a safety case can rest on and one that collapses the first time somebody checks.

`growth_ratio` and `rise` are pure over a sequence of boxes, so both rules are testable against constructed
approaches rather than by hunting for a real one and hoping.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("signal_approach")

# How much bigger the signal has to get before the gap is meaningfully closing. Below this it is box jitter
# on a stationary vehicle at a light, which is the commonest thing a dashcam records at a junction.
MIN_GROWTH_RATIO = 1.35
# Frames the approach must span. Two boxes can grow by chance; a sustained trend cannot.
MIN_FRAMES = 4
# A signal overhead of the ego drifts down the image as it is approached. Required so a signal merely
# resolving better at distance, which also grows, is not read as an approach.
MIN_RISE_FRAC = 0.02
# States that make an approach worth reporting. Amber is included: continuing to close on an amber is the
# decision a planner is judged on, and excluding it would answer only the easy half of the question.
STOP_STATES = ("R", "Y")


def _median(xs: list[float]) -> float:
    """Median of a short list.

    A median rather than a mean at both ends of both measures. A detector that loses the signal for one
    frame emits a box at the origin, and a mean over three samples is moved a long way by one of those,
    which was enough to invert the direction of travel and turn a real approach into a refusal.
    """
    if not xs:
        return 0.0
    ordered = sorted(xs)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def growth_ratio(boxes: list) -> float:
    """How much the box grew from the first third of the run to the last, by area.

    Thirds rather than first-to-last, because a single mis-sized box at either end would otherwise decide
    the answer for the whole approach.
    """
    areas = [max(1.0, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes if b and len(b) >= 4]
    if len(areas) < 2:
        return 1.0
    k = max(1, len(areas) // 3)
    first = _median(areas[:k])
    last = _median(areas[-k:])
    return float(last / first) if first > 0 else 1.0


def rise(boxes: list, frame_height: int) -> float:
    """How far the box centre moved down the frame, as a fraction of frame height.

    Positive means it drifted downward, which is what a signal mounted above the road does as you approach
    it. A signal that only grows without drifting is one being resolved better at distance, not neared.
    """
    ys = [((b[1] + b[3]) / 2.0) for b in boxes if b and len(b) >= 4]
    if len(ys) < 2 or not frame_height:
        return 0.0
    k = max(1, len(ys) // 3)
    return float((_median(ys[-k:]) - _median(ys[:k])) / frame_height)


def classify_approach(boxes: list, frame_height: int) -> tuple[bool, dict]:
    """Whether this run of boxes shows the ego closing on the signal, and the evidence."""
    g = growth_ratio(boxes)
    r = rise(boxes, frame_height)
    ev = {"frames": len(boxes), "growth_ratio": round(g, 3), "rise_frac": round(r, 4),
          "min_growth": MIN_GROWTH_RATIO, "min_frames": MIN_FRAMES, "min_rise": MIN_RISE_FRAC}
    if len(boxes) < MIN_FRAMES:
        ev["reason"] = "too few frames for a trend to mean anything"
        return False, ev
    if g < MIN_GROWTH_RATIO:
        ev["reason"] = ("the signal did not grow enough to be nearing; this is what waiting at a light "
                        "looks like")
        return False, ev
    if r < MIN_RISE_FRAC:
        ev["reason"] = ("the signal grew without drifting down the frame, which is a distant signal "
                        "resolving rather than one being approached")
        return False, ev
    ev["reason"] = "the signal grew and drifted down the frame, so the gap was closing"
    return True, ev


async def detect_signal_approaches(db, session_id) -> list[dict]:
    """Approaches on a stopping signal, for one session.

    Each signal phase already says which light was which colour and when. This asks, for the frames inside a
    red or amber phase, what the same track's boxes were doing.
    """
    import uuid as _uuid

    from sqlalchemy import select

    from db.models import Frame, Object, TimelineEvent

    sid = session_id if isinstance(session_id, _uuid.UUID) else _uuid.UUID(str(session_id))

    phases = (await db.execute(
        select(TimelineEvent).where(
            TimelineEvent.session_id == sid,
            TimelineEvent.kind == "signal_phase",
            TimelineEvent.track_id.isnot(None)))).scalars().all()
    stopping = [p for p in phases if str((p.payload or {}).get("state")) in STOP_STATES]
    if not stopping:
        return []

    track_ids = {p.track_id for p in stopping}
    rows = (await db.execute(
        select(Object.track_id, Object.bbox, Frame.ts_ns, Frame.frame_id, Frame.height)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(Frame.session_id == sid, Object.track_id.in_(track_ids))
        .order_by(Frame.ts_ns))).all()

    by_track: dict = {}
    for tid, bbox, ts, fid, h in rows:
        by_track.setdefault(str(tid), []).append((int(ts or 0), bbox, str(fid), int(h or 0)))

    out: list[dict] = []
    for p in stopping:
        seq = by_track.get(str(p.track_id)) or []
        end = p.t_end_ns if p.t_end_ns is not None else p.t_start_ns
        inside = [(ts, b, fid, h) for ts, b, fid, h in seq if p.t_start_ns <= ts <= end]
        if not inside:
            continue
        boxes = [b for _ts, b, _f, _h in inside]
        height = next((h for _ts, _b, _f, h in inside if h), 0)
        approaching, ev = classify_approach(boxes, height)
        if not approaching:
            continue
        state = str((p.payload or {}).get("state"))
        out.append({
            "kind": "signal_approach_on_red",
            "track_id": str(p.track_id),
            "frame_id": inside[0][2],
            "t_start_ns": inside[0][0],
            "t_end_ns": inside[-1][0],
            # Confidence rises with how decisively the signal grew, capped: a huge ratio usually means the
            # box was tiny at the start, not that the approach is more certain.
            "conf": round(min(0.95, 0.45 + 0.25 * (ev["growth_ratio"] - MIN_GROWTH_RATIO)), 3),
            "payload": {"signal_state": state, **ev,
                        "claim": ("the ego was closing on a signal showing "
                                  f"{state} and kept closing; this is an approach, not a proven entry "
                                  "into the box, which monocular geometry cannot establish")},
        })

    log.info("signal_approach.derived", session=str(sid), phases=len(stopping), found=len(out))
    return out
