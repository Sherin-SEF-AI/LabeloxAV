"""Give lanes an identity across frames.

`Lane.track_ref` is the column that says "this lane in frame 40 is the same lane as that one in frame 39",
and on the real corpus it is null on 4,546 of 4,558 rows. The only thing that has ever set it is the optical
flow propagator, which needs both images decoded per frame pair and has therefore never run outside a
handful of frames somebody stepped through by hand.

That null is why no lane behaviour has ever been derived. A lane change is an actor crossing the same
boundary it was on the other side of a moment ago, and "the same boundary" is precisely what a null track_ref
denies you. Comparing an actor against a lane that exists only in the current frame can never show a
crossing, because there is no previous frame to have been on the other side of.

Linking does not need the images. Two lanes in consecutive frames are the same lane if their curves nearly
coincide over the heights they share, and that is computable from the control points alone, which are already
in Postgres. It is cheap enough to run over a whole session in one pass, which is the difference between a
capability that exists and one that has run.

The matching is greedy on a cost matrix rather than optimal assignment. With the three to six lanes a frame
actually holds, greedy and Hungarian agree, and greedy does not add scipy to the import graph of a module
that runs inside the API process.
"""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass

from core.logging import get_logger
from services.intelligence.lane_events import lane_x_at

log = get_logger("lane_linking")

# How far a lane boundary may drift sideways in the image, per second. A budget per frame pair would be
# wrong: the fleet captures at anything from 3 to 30 fps, and at 3 fps a lane legitimately moves ten times as
# far between consecutive frames as it does at 30. Expressed as a fraction of the frame width per second so
# it holds across capture resolutions as well as rates.
MAX_LATERAL_RATE_FRAC_PER_S = 0.25
# The rate alone would admit anything across a long gap, so it is capped. Past this the two curves are
# further apart than a lane is wide and calling them the same lane is a guess.
MAX_MEAN_DX_FRAC = 0.12
# Curves that share less than this fraction of their height range are not really comparable, and matching on
# a sliver of overlap is how a near lane gets linked to a far one.
MIN_OVERLAP_FRAC = 0.25
# Beyond this many frames without a match, a lane identity is closed rather than bridged. A lane that
# disappears for a second and comes back is usually a different lane by then.
MAX_GAP_FRAMES = 3


@dataclass(frozen=True)
class LaneRow:
    lane_id: str
    frame_id: str
    ts_ns: int
    control_points: list
    lane_type: str
    is_ego: bool


def _y_range(cps: list) -> tuple[float, float] | None:
    ys = [float(p[1]) for p in (cps or []) if p is not None and len(p) >= 2]
    return (min(ys), max(ys)) if len(ys) >= 2 else None


def curve_distance(a: list, b: list, samples: int = 12) -> tuple[float, float] | None:
    """Mean horizontal separation of two lane curves over the heights they share, and the shared fraction.

    Returns None when the curves do not overlap in height at all. Horizontal separation rather than a full
    curve distance because lanes are functions of height in image space, and the vertical component of any
    distance between two lane curves is an artefact of where the control points happened to land.
    """
    ra, rb = _y_range(a), _y_range(b)
    if ra is None or rb is None:
        return None
    lo, hi = max(ra[0], rb[0]), min(ra[1], rb[1])
    if hi <= lo:
        return None
    span = max(ra[1] - ra[0], rb[1] - rb[0])
    overlap_frac = (hi - lo) / span if span > 0 else 0.0

    total, n = 0.0, 0
    for i in range(samples):
        y = lo + (hi - lo) * i / max(1, samples - 1)
        xa, xb = lane_x_at(a, y), lane_x_at(b, y)
        if xa is None or xb is None:
            continue
        total += abs(xa - xb)
        n += 1
    if n == 0:
        return None
    return total / n, overlap_frac


def match_frames(prev: list[LaneRow], cur: list[LaneRow], *, frame_width: int,
                 dt_ns: int | None = None) -> dict[str, str]:
    """Greedy nearest-curve matching between two frames. Returns cur lane_id -> prev lane_id.

    Lane type must agree. A solid line does not become dashed between two frames, and allowing the match
    would let a lane identity change type mid-session, which then makes the illegal-crossing rule depend on
    which frame the crossing happened to be detected in.

    The distance budget comes from the elapsed time, not from a constant. On this corpus consecutive frames
    are 333ms apart and genuine matches sit at 40 to 105 pixels; a fixed budget tight enough for 30fps
    rejects every one of them, which is why the column this feeds has been null on 4,546 of 4,558 rows.
    """
    w = max(1, frame_width)
    dt_s = (dt_ns / 1e9) if dt_ns and dt_ns > 0 else 0.05
    max_dx = min(MAX_MEAN_DX_FRAC * w, MAX_LATERAL_RATE_FRAC_PER_S * w * dt_s)
    costs: list[tuple[float, str, str]] = []
    for c in cur:
        for p in prev:
            if c.lane_type != p.lane_type:
                continue
            d = curve_distance(c.control_points, p.control_points)
            if d is None:
                continue
            mean_dx, overlap = d
            if mean_dx > max_dx or overlap < MIN_OVERLAP_FRAC:
                continue
            costs.append((mean_dx, c.lane_id, p.lane_id))

    costs.sort()
    taken_cur: set[str] = set()
    taken_prev: set[str] = set()
    out: dict[str, str] = {}
    for _cost, cur_id, prev_id in costs:
        if cur_id in taken_cur or prev_id in taken_prev:
            continue
        out[cur_id] = prev_id
        taken_cur.add(cur_id)
        taken_prev.add(prev_id)
    return out


def link_session_lanes(lanes: list[LaneRow], *, frame_width: int) -> dict[str, str]:
    """Assign every lane a cross-frame identity. Returns lane_id -> track_ref.

    A lane that matches nothing starts a new identity rather than being dropped. A single-frame lane is still
    a lane; it just cannot participate in a crossing, and the deriver works that out for itself from the
    length of the series.
    """
    by_frame: dict[str, list[LaneRow]] = {}
    frame_ts: dict[str, int] = {}
    for row in lanes:
        by_frame.setdefault(row.frame_id, []).append(row)
        frame_ts[row.frame_id] = row.ts_ns

    order = sorted(by_frame, key=lambda f: frame_ts[f])
    identity: dict[str, str] = {}
    # Recent frames, newest last, so a lane can bridge a frame where it was not detected.
    recent: list[list[LaneRow]] = []

    for frame_id in order:
        cur = by_frame[frame_id]
        matched: dict[str, str] = {}
        for prev in reversed(recent[-MAX_GAP_FRAMES:]):
            remaining = [c for c in cur if c.lane_id not in matched]
            if not remaining:
                break
            # The budget grows with the real gap, so bridging two frames back allows twice the drift of one.
            dt_ns = max(0, frame_ts[frame_id] - prev[0].ts_ns)
            for cur_id, prev_id in match_frames(prev, remaining, frame_width=frame_width,
                                                dt_ns=dt_ns).items():
                matched[cur_id] = identity[prev_id]

        for c in cur:
            identity[c.lane_id] = matched.get(c.lane_id) or str(_uuid.uuid4())
        recent.append(cur)

    return identity


async def link_lanes_for_session(db, session_id, *, apply: bool = True) -> dict:
    """Link one session's lanes and write the identities back.

    Lanes that already carry a track_ref keep it. Something set it deliberately, either the optical flow
    propagator or an import that brought identities with it, and geometry has no business overruling either.
    """
    from sqlalchemy import select

    from db.models import Frame, Lane

    sid = session_id if isinstance(session_id, _uuid.UUID) else _uuid.UUID(str(session_id))
    rows = (await db.execute(
        select(Lane, Frame.ts_ns, Frame.width)
        .join(Frame, Lane.frame_id == Frame.frame_id)
        .where(Lane.session_id == sid).order_by(Frame.ts_ns))).all()
    if not rows:
        return {"session_id": str(sid), "lanes": 0, "detail": "the session has no lanes"}

    width = max((int(w or 0) for _l, _ts, w in rows), default=1280) or 1280
    already = {str(lane.lane_id) for lane, _ts, _w in rows if lane.track_ref is not None}
    unlinked = [LaneRow(lane_id=str(lane.lane_id), frame_id=str(lane.frame_id), ts_ns=int(ts or 0),
                        control_points=lane.control_points, lane_type=str(lane.lane_type),
                        is_ego=bool(lane.is_ego))
                for lane, ts, _w in rows if str(lane.lane_id) not in already]

    identity = link_session_lanes(unlinked, frame_width=width)
    distinct = len(set(identity.values()))
    multi = sum(1 for ref in set(identity.values())
                if sum(1 for v in identity.values() if v == ref) > 1)

    written = 0
    if apply:
        by_id = {str(lane.lane_id): lane for lane, _ts, _w in rows}
        for lane_id, ref in identity.items():
            lane = by_id.get(lane_id)
            if lane is None:
                continue
            lane.track_ref = _uuid.UUID(ref)
            written += 1
        await db.commit()

    log.info("lane_linking.done", session=str(sid), lanes=len(rows), linked=written,
             identities=distinct, multi_frame=multi)
    return {"session_id": str(sid), "lanes": len(rows), "already_linked": len(already),
            "linked": written, "identities": distinct,
            # The number that says whether linking achieved anything. Identities that appear in one frame
            # only cannot support a crossing, so a session where every identity is single-frame has been
            # linked in name and not in fact.
            "multi_frame_identities": multi, "frame_width": width, "dry_run": not apply}
