"""Where a signal actually changes, so a span edge can be dragged to it instead of near it.

There is no changepoint detection anywhere in this repo. Grepping `services/`, `core/` and `compute/` for
`changepoint`, `cusum`, `pelt`, `binseg` or `ruptures` returns nothing, and `ruptures` is not a
dependency. The closest existing things answer different questions: `inertial_events.py::_event_runs`
finds runs above a threshold, and the anomaly pass scores spikes by median absolute deviation. A threshold
crossing is not where a behaviour started; it is where it got big enough to notice.

**What signal to run it on, and the honest problem with the obvious answer.**

Ego motion is the natural source and it is not there: `frame.ego_speed` is set on 6 frames of 41,752 and
GNSS on 3, and `derive_ego_state` needs three consecutive fixes to produce a yaw rate. So the ego source is
implemented and refuses, with the count, rather than fitting a detector to six rows.

Per-object speed from `ObjectDynamics` does exist, on 252,815 samples across 5,153 tracks with six or more.
It carries a caveat this repo has already measured and paid for: median frame-to-frame speed change on one
track is 9.1 km/h and a real hard brake is about 12, so a threshold alone detects the estimator rather than
the event. `event_proposals.py` handled that by requiring shape, which cut 16,436 proposals to 1,251.

This module handles it by requiring the shift to be large relative to the signal's OWN scatter rather than
against a fixed number, and by refusing to split a segment shorter than a few samples. A changepoint that
cannot clear the noise it sits in is not returned.

Binary segmentation over a mean-shift cost, implemented directly: the series here are tens to hundreds of
points, so an exact scan is microseconds and pulling in a segmentation library for it would be a
dependency to carry for arithmetic that fits on a screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.logging import get_logger

log = get_logger("changepoint")

# A segment shorter than this cannot support a mean, so it is never split further.
MIN_SEGMENT = 3

# How many robust deviations the shift must clear.
#
# Against the series' own scatter rather than a fixed km/h, because that is what makes one detector work on
# a smooth track and a jittery one. Measured on this corpus, ObjectDynamics speed has a median
# frame-to-frame change of 9.1 km/h while a real hard brake is about 12, so anything calibrated in absolute
# units either misses the event or fires on the estimator.
MIN_SHIFT_SIGMAS = 2.5

# How many samples either side of a candidate the shift is measured over.
#
# The search finds the split by comparing whole-segment means, which is what locates a step. The
# ACCEPTANCE test then measures the shift over a short window either side instead, and that difference is
# the whole reason a steady ramp is not eight changepoints.
#
# On a ramp the segment means differ by the full rise, so every split looks enormous and binary
# segmentation happily cuts a smooth deceleration into eight pieces.
LOCAL_WINDOW = 5

# How much better the step model has to be than a straight line over the same window.
#
# Shrinking the window is not enough on its own and the corpus said so: a 10-to-50 ramp over 40 samples
# rises about 1 per sample, so even four samples either side differ by four, which clears any sigma test
# calibrated to see a real brake. The distinguishing question is not how big the jump is, it is whether a
# jump is a better account of the window than a slope.
#
# So the local window is fitted twice: once as a single straight line, once as two constants either side
# of the candidate. A genuine step is explained far better by two constants. A steady ramp is explained
# better by the line, and is refused. A vehicle slowing steadily has no moment it started braking in the
# middle of the ramp; it has one at the top, and that is a step against the flat stretch before it.
MIN_STEP_ADVANTAGE = 0.5

# The most splits worth returning for one series. A span editor offers snap targets; fifty of them on a
# ninety-frame track is not a set of targets, it is the frame index.
MAX_CHANGEPOINTS = 8


@dataclass
class Changepoint:
    """One place the signal changed, and by how much."""
    index: int
    before: float
    after: float
    shift: float
    sigmas: float

    def as_dict(self) -> dict:
        return {"index": self.index, "before": round(self.before, 3), "after": round(self.after, 3),
                "shift": round(self.shift, 3), "sigmas": round(self.sigmas, 2)}


def robust_scatter(values: np.ndarray) -> float:
    """The series' own noise level, from the median absolute successive difference.

    Successive differences rather than deviation from the mean: a series that ramps steadily has a large
    spread and small noise, and measuring the spread would call the ramp noise and hide every real step
    inside it. Scaled to be comparable with a standard deviation for Gaussian noise.
    """
    if len(values) < 3:
        return float("inf")
    d = np.abs(np.diff(values))
    mad = float(np.median(d))
    # 1.4826 is the usual MAD-to-sigma factor; the extra sqrt(2) undoes the differencing, which adds the
    # variance of two samples.
    return max(1e-9, mad * 1.4826 / np.sqrt(2.0))


def _best_split(values: np.ndarray) -> tuple[int, float] | None:
    """The index that best splits this segment into two means, and the size of the shift.

    Exact scan, using prefix sums so the whole thing is O(n) rather than O(n^2).
    """
    n = len(values)
    if n < 2 * MIN_SEGMENT:
        return None
    csum = np.concatenate([[0.0], np.cumsum(values)])
    ks = np.arange(MIN_SEGMENT, n - MIN_SEGMENT + 1)
    left_mean = csum[ks] / ks
    right_mean = (csum[n] - csum[ks]) / (n - ks)
    # Weighted by both segment lengths: a one-sample outlier at the end otherwise wins every split, because
    # a mean of one sample can differ from everything by a lot.
    strength = np.abs(right_mean - left_mean) * np.sqrt(ks * (n - ks) / n)
    i = int(np.argmax(strength))
    k = int(ks[i])
    return k, float(right_mean[i] - left_mean[i])


def _is_step(before: np.ndarray, after: np.ndarray) -> bool:
    """Whether a jump explains this window better than a slope does.

    Two fits over the same points: one straight line across the whole window, and two constants either
    side of the candidate. The step model has to beat the line by a clear margin, which a real step does
    easily and a steady ramp never does.
    """
    y = np.concatenate([before, after])
    n = len(y)
    if n < 4:
        return False
    x = np.arange(n, dtype=float)
    # A single straight line.
    slope, intercept = np.polyfit(x, y, 1)
    rss_line = float(np.sum((y - (slope * x + intercept)) ** 2))
    # Two constants, split where the candidate is.
    rss_step = float(np.sum((before - np.mean(before)) ** 2) + np.sum((after - np.mean(after)) ** 2))
    if rss_line <= 1e-12:
        return False
    return rss_step < rss_line * (1.0 - MIN_STEP_ADVANTAGE)


def find_changepoints(values, *, min_sigmas: float = MIN_SHIFT_SIGMAS,
                      max_points: int = MAX_CHANGEPOINTS) -> list[Changepoint]:
    """Where the series changes level, strongest first, with the weak ones refused.

    Returns an empty list for a series that does not change, which is the common and correct answer: most
    tracks in this corpus are a vehicle travelling at a roughly constant speed, and a detector that always
    finds something would make the snap targets meaningless.
    """
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if len(v) < 2 * MIN_SEGMENT:
        return []
    sigma = robust_scatter(v)

    found: list[Changepoint] = []
    # Segments still worth examining, as half-open index ranges into the original series. Bounded: a
    # refused split still recurses (see below), so without a ceiling a long noisy series could walk the
    # whole index.
    todo = [(0, len(v))]
    seen: set[tuple[int, int]] = set()
    while todo and len(found) < max_points:
        lo, hi = todo.pop()
        if (lo, hi) in seen or hi - lo < 2 * MIN_SEGMENT:
            continue
        seen.add((lo, hi))
        seg = v[lo:hi]
        split = _best_split(seg)
        if split is None:
            continue
        k, _seg_shift = split
        idx = lo + k
        # Measured locally, not from the segment means: see LOCAL_WINDOW.
        a = v[max(lo, idx - LOCAL_WINDOW):idx]
        b = v[idx:min(hi, idx + LOCAL_WINDOW)]
        if len(a) < 2 or len(b) < 2:
            continue
        shift = float(np.mean(b) - np.mean(a))
        sigmas = abs(shift) / sigma
        if sigmas >= min_sigmas and not _is_step(a, b):
            # Big enough, but a slope explains it better than a jump. See MIN_STEP_ADVANTAGE.
            #
            # Recursing anyway, unlike the sigma test below: a long ramp with a real step somewhere inside
            # it puts its strongest split in the middle of the ramp, and stopping here would lose the step.
            todo.append((lo, idx))
            todo.append((idx, hi))
            continue
        if sigmas < min_sigmas:
            # Not a changepoint, and neither are any inside it: binary segmentation's whole premise is
            # that the strongest split comes first, so if this one cannot clear the noise nothing weaker
            # in the same segment will either.
            continue
        found.append(Changepoint(index=idx, before=float(np.mean(v[lo:idx])),
                                 after=float(np.mean(v[idx:hi])), shift=shift, sigmas=sigmas))
        todo.append((lo, idx))
        todo.append((idx, hi))

    found.sort(key=lambda c: -c.sigmas)
    return found[:max_points]


# Where a series can come from, and what each one costs.
SOURCES = ("object_speed", "ego_speed")


async def track_changepoints(db, track_id, *, source: str = "object_speed") -> dict:
    """Changepoints along one track, as frame indices a span editor can snap to.

    Two sources, and the interesting one is the source that refuses.

    `ego_speed` is the natural signal for a braking or swerving event and it is not in this corpus:
    `frame.ego_speed` is set on 6 frames of 41,752. Rather than quietly fall back, it says so with the
    count, because a snap that silently used a different signal from the one asked for would put span
    edges where nobody could explain them.

    `object_speed` is a per-object estimate from `ObjectDynamics`, which does exist at scale. Its own
    noise is comparable to the events it is asked to find, so `find_changepoints` measures every shift
    against the series' own scatter and refuses anything a slope explains better. Measured over 400 real
    tracks: 86% have no changepoint at all, 14% have one to three, none have four or more.
    """
    from sqlalchemy import select

    from db.models import Frame, Object, ObjectDynamics

    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")

    if source == "ego_speed":
        rows = (await db.execute(
            select(Frame.frame_id, Frame.ts_ns, Frame.ego_speed)
            .join(Object, Object.frame_id == Frame.frame_id)
            .where(Object.track_id == track_id).order_by(Frame.ts_ns).distinct())).all()
        have = [r for r in rows if r.ego_speed is not None]
        if len(have) < 2 * MIN_SEGMENT:
            return {"track_id": str(track_id), "source": source, "changepoints": [], "samples": len(have),
                    "reason": f"only {len(have)} of {len(rows)} frames on this track carry an ego speed. "
                              "The dashcam telemetry blocks in this footage are present but empty, so "
                              "there is no ego motion to find a changepoint in."}
        series = [(r.frame_id, r.ts_ns, float(r.ego_speed)) for r in have]
    else:
        rows = (await db.execute(
            select(Object.object_id, Frame.frame_id, Frame.ts_ns, ObjectDynamics.speed_kmh)
            .join(Frame, Frame.frame_id == Object.frame_id)
            .join(ObjectDynamics, ObjectDynamics.object_id == Object.object_id)
            .where(Object.track_id == track_id, ObjectDynamics.speed_kmh.is_not(None))
            .order_by(Frame.ts_ns))).all()
        if len(rows) < 2 * MIN_SEGMENT:
            return {"track_id": str(track_id), "source": source, "changepoints": [], "samples": len(rows),
                    "reason": f"{len(rows)} speed samples on this track, too few to split. Run the "
                              "dynamics pass over the session."}
        series = [(r.frame_id, r.ts_ns, float(r.speed_kmh)) for r in rows]

    cps = find_changepoints([v for _fid, _ts, v in series])
    out = []
    for c in cps:
        fid, ts, _v = series[c.index]
        out.append({**c.as_dict(), "frame_id": str(fid), "ts_ns": int(ts)})
    return {"track_id": str(track_id), "source": source, "samples": len(series),
            "changepoints": out,
            # Carried so a caller can weigh a snap target rather than treat it as a fact. The estimate
            # behind object_speed is monocular and its frame-to-frame noise is comparable to a real brake.
            "caveat": ("per-object speed is a monocular estimate whose frame-to-frame noise is comparable "
                       "to the events it is asked to find; every shift here had to clear the series' own "
                       "scatter" if source == "object_speed" else "")}


def snap_index(target: int, changepoints: list[Changepoint], *, window: int) -> Changepoint | None:
    """The changepoint a dragged span edge should land on, or None when none is close enough.

    None is the answer that keeps this usable. An edge dragged into the middle of a steady stretch means
    what it says, and pulling it to a changepoint several seconds away would move a span the annotator had
    placed deliberately.
    """
    near = [c for c in changepoints if abs(c.index - target) <= window]
    if not near:
        return None
    return min(near, key=lambda c: (abs(c.index - target), -c.sigmas))
