"""Lane topology: pair boundaries into lanelets and chain the lanelets that follow one another.

This is the layer the exports were missing, and without it neither export was a map. A Lanelet2 file is
defined by `relation type=lanelet` members naming a left and a right boundary; the exporter emitted the
boundaries as loose ways and no relations, so nothing downstream could route on it. OpenDRIVE roads were
emitted with a centre lane of `type="none"` and no driving lanes at all, which is geometry with no road
in it. Both formats need the same two facts that a pile of polylines does not carry: which boundaries
bound the same lane, and which lane follows which.

Everything here is pure geometry over polylines so it can be tested without a database, and every
boundary that cannot be paired is returned in `unpaired` with the reason. A lane invented to make the
output look complete is worse than a boundary reported as unpaired, because the first is indistinguishable
from a real one downstream.

Distances are computed in a local tangent plane about the map's own centre. Over a single drive the
approximation is far below the metre-scale tolerances that matter here, and it keeps the module free of a
projection dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

M_PER_DEG_LAT = 111320.0

# What counts as one lane's worth of separation. The upper bound is deliberately generous for Indian
# roads, where a marked lane is often wider than the 3.0 to 3.5 m a European map assumes and where the
# outer boundary is frequently the road edge rather than a marking.
MIN_LANE_W_M = 2.0
MAX_LANE_W_M = 5.5

# Two boundaries bound the same lane only if they run the same way. Twenty degrees tolerates a bend taken
# at slightly different radii by two independently traced boundaries without admitting a cross street.
MAX_HEADING_DIFF_RAD = math.radians(20.0)

# How much of the shorter boundary must sit alongside the longer one. Below this the two are more likely
# to be consecutive stretches of the same boundary than the two sides of one lane.
MIN_OVERLAP = 0.5

# End-to-start distance within which one lanelet is taken to feed the next.
SUCCESSOR_TOL_M = 4.0


@dataclass(frozen=True)
class Lanelet:
    """One drivable lane: its two boundaries, the centreline between them, and where it came from."""

    left: tuple[tuple[float, float], ...]      # (lon, lat), in travel order
    right: tuple[tuple[float, float], ...]
    centre: tuple[tuple[float, float], ...]
    width_m: float
    left_type: str
    right_type: str
    confidence: float
    source_elements: tuple[str, ...] = field(default=())


def _origin(polylines: list[list[tuple[float, float]]]) -> tuple[float, float]:
    pts = [p for line in polylines for p in line]
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def to_local(line: list[tuple[float, float]], origin: tuple[float, float]) -> list[tuple[float, float]]:
    """(lon, lat) degrees to (east, north) metres about `origin`."""
    olon, olat = origin
    mlon = M_PER_DEG_LAT * math.cos(math.radians(olat))
    return [((lon - olon) * mlon, (lat - olat) * M_PER_DEG_LAT) for lon, lat in line]


def to_world(line: list[tuple[float, float]], origin: tuple[float, float]) -> list[tuple[float, float]]:
    olon, olat = origin
    mlon = M_PER_DEG_LAT * math.cos(math.radians(olat))
    return [(olon + e / mlon, olat + n / M_PER_DEG_LAT) for e, n in line]


def arclength(line: list[tuple[float, float]]) -> float:
    return sum(math.dist(line[i], line[i + 1]) for i in range(len(line) - 1))


def resample(line: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """`n` points evenly spaced along the polyline by arclength.

    Boundaries are traced independently and share no vertices, so comparing them vertex to vertex
    compares whatever spacing each happened to get. Resampling both to a common count is what makes a
    lateral offset between them mean anything.
    """
    if n < 2 or len(line) < 2:
        return list(line)
    segs = [math.dist(line[i], line[i + 1]) for i in range(len(line) - 1)]
    total = sum(segs)
    if total <= 0:
        return [line[0]] * n
    out, target, acc, i = [line[0]], total / (n - 1), 0.0, 0
    for k in range(1, n - 1):
        want = k * target
        while i < len(segs) and acc + segs[i] < want:
            acc += segs[i]
            i += 1
        if i >= len(segs):
            out.append(line[-1])
            continue
        t = (want - acc) / segs[i] if segs[i] > 0 else 0.0
        (x0, y0), (x1, y1) = line[i], line[i + 1]
        out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    out.append(line[-1])
    return out


def heading(line: list[tuple[float, float]]) -> float:
    """Overall direction of travel, end to end, in radians east-of-north."""
    (x0, y0), (x1, y1) = line[0], line[-1]
    return math.atan2(x1 - x0, y1 - y0)


def _angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


def signed_offset(a: list[tuple[float, float]], b: list[tuple[float, float]], n: int = 12) -> tuple[float, float]:
    """Mean lateral offset of `b` from `a` and the fraction of samples that agree on its sign.

    The sign says which side `b` lies on, which is what decides left from right. The agreement fraction
    is the guard against pairing two boundaries that cross: a real pair keeps `b` on one side for its
    whole length, while a crossing pair changes sign part way along and would otherwise pass a test that
    only looked at the mean.
    """
    ra, rb = resample(a, n), resample(b, n)
    offs = []
    for i in range(n):
        j = min(i + 1, n - 1)
        k = max(i - 1, 0)
        tx, ty = ra[j][0] - ra[k][0], ra[j][1] - ra[k][1]
        norm = math.hypot(tx, ty)
        if norm <= 0:
            continue
        # left-hand normal of the tangent
        nx, ny = -ty / norm, tx / norm
        dx, dy = rb[i][0] - ra[i][0], rb[i][1] - ra[i][1]
        offs.append(dx * nx + dy * ny)
    if not offs:
        return (0.0, 0.0)
    mean = sum(offs) / len(offs)
    agree = sum(1 for o in offs if (o > 0) == (mean > 0)) / len(offs)
    return (mean, agree)


def overlap_fraction(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """How much of the shorter boundary lies alongside the longer one, by projection onto its direction."""
    la, lb = arclength(a), arclength(b)
    if min(la, lb) <= 0:
        return 0.0
    h = heading(a)
    ux, uy = math.sin(h), math.cos(h)

    def span(line):
        ts = [p[0] * ux + p[1] * uy for p in line]
        return min(ts), max(ts)

    a0, a1 = span(a)
    b0, b1 = span(b)
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(1e-9, min(a1 - a0, b1 - b0))


def _centreline(left: list[tuple[float, float]], right: list[tuple[float, float]],
                n: int = 24) -> list[tuple[float, float]]:
    la, ra = resample(left, n), resample(right, n)
    return [((la[i][0] + ra[i][0]) / 2.0, (la[i][1] + ra[i][1]) / 2.0) for i in range(n)]


def build_lanelets(boundaries: list[dict], *, min_w: float = MIN_LANE_W_M, max_w: float = MAX_LANE_W_M,
                   ) -> dict:
    """Pair boundaries into lanelets, then chain the lanelets that follow one another.

    `boundaries` are dicts with `points` as (lon, lat) in travel order, plus optional `id`, `lane_type`
    and `confidence`. A boundary may serve two lanelets, as the right of one lane and the left of the
    next; that is how adjacent lanes share a marking, so reuse is allowed while duplicate lanelets are
    not.

    Pairing is nearest-valid rather than globally optimal. A global assignment would be the better answer
    on a wide multi-lane road, and is the upgrade seam here, but it needs more boundaries than a single
    drive of a two-lane street produces and would be untestable on this corpus today.
    """
    lines = [list(b["points"]) for b in boundaries if len(b.get("points") or ()) >= 2]
    keep = [b for b in boundaries if len(b.get("points") or ()) >= 2]
    dropped = [{"id": b.get("id"), "reason": "fewer than two points"}
               for b in boundaries if len(b.get("points") or ()) < 2]
    if len(lines) < 2:
        return {"lanelets": [], "successors": [], "unpaired": dropped + [
            {"id": b.get("id"), "reason": "no other boundary to pair with"} for b in keep],
            "origin": _origin(lines)}

    origin = _origin(lines)
    local = [to_local(line, origin) for line in lines]
    heads = [heading(line) for line in local]

    pairs: dict[tuple[int, int], dict] = {}
    reasons: dict[int, str] = {}
    for i in range(len(local)):
        best = None
        why = "no boundary within a lane's width running the same way"
        for j in range(len(local)):
            if i == j:
                continue
            if _angle_diff(heads[i], heads[j]) > MAX_HEADING_DIFF_RAD:
                continue
            ov = overlap_fraction(local[i], local[j])
            if ov < MIN_OVERLAP:
                why = "the nearest parallel boundary runs alongside for too little of its length"
                continue
            off, agree = signed_offset(local[i], local[j])
            w = abs(off)
            if not (min_w <= w <= max_w):
                why = f"the nearest parallel boundary is {w:.1f} m away, outside one lane's width"
                continue
            if agree < 0.9:
                why = "the nearest parallel boundary crosses this one rather than running beside it"
                continue
            if best is None or w < best[1]:
                best = (j, w, off, ov)
        if best is None:
            reasons[i] = why
            continue
        j, w, off, ov = best
        # `off` is the offset of j from i along i's left normal, so a positive offset puts j on the left.
        left, right = (j, i) if off > 0 else (i, j)
        pairs.setdefault((left, right), {"width_m": w, "overlap": ov})

    lanelets: list[Lanelet] = []
    used: set[int] = set()
    for (li, ri), meta in sorted(pairs.items()):
        lc = _centreline(local[li], local[ri])
        lanelets.append(Lanelet(
            left=tuple(to_world(resample(local[li], 24), origin)),
            right=tuple(to_world(resample(local[ri], 24), origin)),
            centre=tuple(to_world(lc, origin)),
            width_m=round(meta["width_m"], 3),
            left_type=str(keep[li].get("lane_type") or "unknown"),
            right_type=str(keep[ri].get("lane_type") or "unknown"),
            # A lane is bounded by two traces and is no better placed than the worse of them.
            confidence=round(min(float(keep[li].get("confidence") or 0.0),
                                 float(keep[ri].get("confidence") or 0.0)), 4),
            source_elements=tuple(str(x) for x in (keep[li].get("id"), keep[ri].get("id")) if x),
        ))
        used.update((li, ri))

    unpaired = dropped + [{"id": keep[i].get("id"),
                           "reason": reasons.get(i, "paired boundary was claimed by another lane")}
                          for i in range(len(keep)) if i not in used]
    return {"lanelets": lanelets, "successors": chain(lanelets), "unpaired": unpaired, "origin": origin}


def chain(lanelets: list[Lanelet], tol_m: float = SUCCESSOR_TOL_M) -> list[tuple[int, int]]:
    """Which lanelet feeds which, as index pairs.

    Two conditions, both needed. The end of one centreline must be within `tol_m` of the start of the
    next, and the two must be heading the same way. Distance alone would join a lane to the oncoming lane
    beside it wherever the two ends happen to meet, which at a junction they routinely do.
    """
    if len(lanelets) < 2:
        return []
    origin = _origin([list(ll.centre) for ll in lanelets])
    centres = [to_local(list(ll.centre), origin) for ll in lanelets]
    out = []
    for i, ci in enumerate(centres):
        for j, cj in enumerate(centres):
            if i == j:
                continue
            if math.dist(ci[-1], cj[0]) > tol_m:
                continue
            if _angle_diff(heading(ci), heading(cj)) > MAX_HEADING_DIFF_RAD:
                continue
            out.append((i, j))
    return sorted(out)
