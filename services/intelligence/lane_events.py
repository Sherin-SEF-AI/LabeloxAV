"""Lane behaviour, derived from lane geometry the corpus already carries.

The corpus has 4,558 lane rows and 2,525 tracked objects and has never once recorded that a vehicle changed
lane. Everything needed was there: a lane is a curve in image coordinates with an identity across frames
(`track_ref`), and a track is the same actor across frames. A lane change is the one crossing the other.

Position uses the box bottom-centre as the ground contact point, the same proxy `services/intelligence/
trajectory.py` already uses, and for the same reason: without calibration it is the only point on a 2D box
that approximates where the actor actually touches the road, and it is where a lane boundary is meaningful.
The alternative, the box centre, moves when the box grows and would report a lane change every time a vehicle
approached the camera.

Everything above `_ObsSeries` is pure over sequences of numbers, so the decision rules can be tested with
constructed crossings rather than by finding a real one and hoping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.logging import get_logger
from services.intelligence.event_taxonomy import lane_params

log = get_logger("lane_events")


def lane_x_at(control_points: list, y: float) -> float | None:
    """Where the lane curve sits horizontally at height y, or None if the lane does not reach that height.

    Linear interpolation between control points rather than evaluating the fitted spline. The control points
    are dense enough that the difference is well under a pixel, and interpolating means this function has no
    opinion about which curve family fitted them, so a lane drawn by hand and a lane fitted by the proposer
    are read identically.
    """
    pts = [(float(p[0]), float(p[1])) for p in (control_points or []) if p is not None and len(p) >= 2]
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: p[1])
    if y < pts[0][1] or y > pts[-1][1]:
        # Extrapolating a lane past its last control point invents road that was never annotated, and the
        # invented part is exactly where a distant vehicle would be judged to have crossed it.
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        if y0 <= y <= y1:
            if y1 == y0:
                return (x0 + x1) / 2.0
            t = (y - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    return None


def signed_offset(ground_xy: tuple[float, float], control_points: list) -> float | None:
    """How far the contact point is to the right (positive) or left (negative) of the lane, in pixels."""
    lx = lane_x_at(control_points, ground_xy[1])
    return None if lx is None else float(ground_xy[0]) - lx


@dataclass
class Observation:
    """One (frame, actor, lane) sample: when, where, and which side."""

    ts_ns: int
    frame_id: str
    offset: float


@dataclass
class _Crossing:
    t_start_ns: int
    t_end_ns: int
    from_side: str
    to_side: str
    frames: int
    start_frame_id: str
    end_frame_id: str


def _side(offset: float) -> str:
    return "right" if offset >= 0 else "left"


def find_crossings(obs: list[Observation], *, commit_offset: float,
                   min_frames_after: int) -> list[_Crossing]:
    """Sign changes in the offset series that the actor committed to.

    A crossing is not the frame the sign flipped. It is a flip followed by `min_frames_after` observations
    that stay on the new side and reach `commit_offset` away from the boundary. Without that, every jitter in
    a box edge or a lane fit near the line reads as a lane change, and on a real session the noise outnumbers
    the signal by roughly an order of magnitude.
    """
    if len(obs) < min_frames_after + 1:
        return []

    out: list[_Crossing] = []
    for i in range(1, len(obs)):
        prev, cur = obs[i - 1], obs[i]
        if _side(prev.offset) == _side(cur.offset):
            continue
        after = obs[i:i + min_frames_after]
        if len(after) < min_frames_after:
            break  # the series ends before the crossing could be confirmed, so it is not one
        new_side = _side(cur.offset)
        if any(_side(o.offset) != new_side for o in after):
            continue
        if max(abs(o.offset) for o in after) < commit_offset:
            continue
        out.append(_Crossing(t_start_ns=prev.ts_ns, t_end_ns=after[-1].ts_ns,
                             from_side=_side(prev.offset), to_side=new_side,
                             frames=1 + len(after),
                             start_frame_id=prev.frame_id, end_frame_id=after[-1].frame_id))
    return out


def find_straddle(obs: list[Observation], *, band: float, min_frames: int) -> list[dict]:
    """Runs where the contact point sat inside a narrow band around the boundary without committing."""
    runs: list[dict] = []
    cur: list[Observation] = []
    for o in obs:
        if abs(o.offset) <= band:
            cur.append(o)
            continue
        if len(cur) >= min_frames:
            runs.append(_straddle_run(cur))
        cur = []
    if len(cur) >= min_frames:
        runs.append(_straddle_run(cur))
    return runs


def _straddle_run(run: list[Observation]) -> dict:
    return {"t_start_ns": run[0].ts_ns, "t_end_ns": run[-1].ts_ns, "frames": len(run),
            "mean_abs_offset_px": round(sum(abs(o.offset) for o in run) / len(run), 2),
            "start_frame_id": run[0].frame_id}


def find_weave(crossings: list[_Crossing], *, window_ns: int) -> list[dict]:
    """Crossings of one boundary that reverse inside a short window.

    The actor ended where it started, which is what separates weaving from moving over. Reported per pair
    rather than per crossing so a long slalom produces a sequence of findings a reviewer can step through.
    """
    out: list[dict] = []
    for a, b in zip(crossings, crossings[1:], strict=False):
        if b.to_side == a.from_side and (b.t_end_ns - a.t_start_ns) <= window_ns:
            out.append({"t_start_ns": a.t_start_ns, "t_end_ns": b.t_end_ns, "crossings": 2,
                        "window_ns": b.t_end_ns - a.t_start_ns, "start_frame_id": a.start_frame_id})
    return out


@dataclass
class _ObsSeries:
    lane_type: str
    lane_id: str
    is_ego: bool
    obs: list[Observation] = field(default_factory=list)


def derive_lane_events(series_by_pair: dict[tuple[str, str], _ObsSeries], *,
                       frame_width: int, params: dict | None = None) -> list[dict]:
    """Turn per (track, lane boundary) offset series into candidate events.

    Pure so the rules can be exercised with constructed series. The caller does the database work and knows
    nothing about the geometry; this knows the geometry and nothing about the database.
    """
    p = {**lane_params(), **(params or {})}
    commit = float(p.get("commit_offset_frac", 0.012)) * max(1, frame_width)
    band = float(p.get("straddle_band_frac", 0.008)) * max(1, frame_width)
    min_after = int(p.get("min_frames_after_cross", 3))
    weave_window = int(p.get("weave_window_ns", 4_000_000_000))
    straddle_min = int(p.get("straddle_min_frames", 6))
    illegal = set(p.get("illegal_to_cross") or [])

    events: list[dict] = []
    for (track_id, _lane_ref), s in series_by_pair.items():
        obs = sorted(s.obs, key=lambda o: o.ts_ns)
        crossings = find_crossings(obs, commit_offset=commit, min_frames_after=min_after)

        weaves = find_weave(crossings, window_ns=weave_window)
        # A crossing that is part of a weave is reported as the weave, not twice. Weaving is the finding; the
        # two crossings that make it up are how it was detected, and listing all three would triple-count one
        # behaviour in every rate the mining surfaces compute.
        spanned = set()
        for w in weaves:
            for c in crossings:
                if w["t_start_ns"] <= c.t_start_ns and c.t_end_ns <= w["t_end_ns"]:
                    spanned.add(id(c))

        for c in crossings:
            if id(c) in spanned:
                continue
            kind = "lane_change_illegal" if s.lane_type in illegal else "lane_change"
            direction = "right" if c.to_side == "right" else "left"
            events.append({
                "kind": kind, "track_id": track_id, "frame_id": c.start_frame_id,
                "t_start_ns": c.t_start_ns, "t_end_ns": c.t_end_ns,
                # Confidence is how far past the line the actor committed, capped. A crossing that barely
                # cleared the threshold is a weaker claim than one that ended a full lane away, and the
                # review queue should see the weak ones first.
                "conf": round(min(1.0, 0.5 + 0.5 * (c.frames - min_after) / max(1, min_after)), 3),
                "payload": {"direction": direction, "lane_type": s.lane_type, "lane_id": s.lane_id,
                            "from_side": c.from_side, "to_side": c.to_side, "frames": c.frames,
                            "is_ego_lane": s.is_ego},
            })

        for w in weaves:
            events.append({
                "kind": "lane_weave", "track_id": track_id, "frame_id": w["start_frame_id"],
                "t_start_ns": w["t_start_ns"], "t_end_ns": w["t_end_ns"], "conf": 0.6,
                "payload": {"crossings": w["crossings"], "lane_id": s.lane_id,
                            "lane_type": s.lane_type, "window_ns": w["window_ns"]},
            })

        for st in find_straddle(obs, band=band, min_frames=straddle_min):
            events.append({
                "kind": "lane_straddle", "track_id": track_id, "frame_id": st["start_frame_id"],
                "t_start_ns": st["t_start_ns"], "t_end_ns": st["t_end_ns"],
                "conf": round(min(1.0, st["frames"] / (2.0 * straddle_min)), 3),
                "payload": {"lane_id": s.lane_id, "lane_type": s.lane_type, "frames": st["frames"],
                            "mean_abs_offset_px": st["mean_abs_offset_px"]},
            })

    return events


async def build_series(db, session_id) -> tuple[dict[tuple[str, str], _ObsSeries], int]:
    """Read one session and pair every tracked actor with every lane boundary it was measurable against.

    Lanes are grouped by `track_ref`, the lane's identity across frames. On the real corpus that column is
    almost entirely null, so an unlinked session is linked first: without it every lane exists in one frame
    only, no actor can be observed on both sides of anything, and the deriver returns nothing forever while
    appearing to work.
    """
    import uuid as _uuid

    from sqlalchemy import select

    from db.models import Frame, Lane, Object

    sid = session_id if isinstance(session_id, _uuid.UUID) else _uuid.UUID(str(session_id))

    lane_rows = (await db.execute(
        select(Lane.lane_id, Lane.frame_id, Lane.track_ref, Lane.control_points,
               Lane.lane_type, Lane.is_ego)
        .where(Lane.session_id == sid, Lane.track_ref.isnot(None)))).all()

    if not lane_rows:
        from services.intelligence.lane_linking import link_lanes_for_session

        linked = await link_lanes_for_session(db, sid, apply=True)
        if not linked.get("linked"):
            return {}, 0
        lane_rows = (await db.execute(
            select(Lane.lane_id, Lane.frame_id, Lane.track_ref, Lane.control_points,
                   Lane.lane_type, Lane.is_ego)
            .where(Lane.session_id == sid, Lane.track_ref.isnot(None)))).all()
    if not lane_rows:
        return {}, 0

    lanes_by_frame: dict[str, list] = {}
    for lane_id, frame_id, track_ref, cps, lane_type, is_ego in lane_rows:
        lanes_by_frame.setdefault(str(frame_id), []).append(
            (str(lane_id), str(track_ref), cps, str(lane_type), bool(is_ego)))

    obj_rows = (await db.execute(
        select(Object.track_id, Object.bbox, Frame.frame_id, Frame.ts_ns, Frame.width)
        .join(Frame, Object.frame_id == Frame.frame_id)
        .where(Frame.session_id == sid, Object.track_id.isnot(None))
        .order_by(Frame.ts_ns))).all()

    series: dict[tuple[str, str], _ObsSeries] = {}
    width = 0
    for track_id, bbox, frame_id, ts_ns, w in obj_rows:
        width = max(width, int(w or 0))
        lanes = lanes_by_frame.get(str(frame_id))
        if not lanes or not bbox or len(bbox) < 4:
            continue
        ground = ((float(bbox[0]) + float(bbox[2])) / 2.0, float(bbox[3]))
        for lane_id, track_ref, cps, lane_type, is_ego in lanes:
            off = signed_offset(ground, cps)
            if off is None:
                continue
            key = (str(track_id), track_ref)
            s = series.get(key)
            if s is None:
                s = series[key] = _ObsSeries(lane_type=lane_type, lane_id=lane_id, is_ego=is_ego)
            s.obs.append(Observation(ts_ns=int(ts_ns or 0), frame_id=str(frame_id), offset=off))

    return series, width


async def detect_lane_events(db, session_id) -> list[dict]:
    """The session-level entry: read, pair, derive. Persistence is the orchestrator's job."""
    series, width = await build_series(db, session_id)
    if not series:
        return []
    events = derive_lane_events(series, frame_width=width or 1280)
    log.info("lane_events.derived", session=str(session_id), pairs=len(series), events=len(events))
    return events
