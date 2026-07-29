"""Is this proposed lane on the road at all?

The lane proposer finds bright linear structure, and a dashcam frame is full of bright linear structure that
is not a lane: the striped hoarding along a construction site, the girders of a flyover, a kerb, a shop
awning. On one real frame it produced six "lanes", every one of them an edge of a blue and white striped
wall, drawn as thick diagonals across the sky. They obliterated the drivable overlay underneath and made a
correct segmentation look broken.

The disqualifying fact is already computed: the drivable mask says where the road is. A lane sits on the
road. A line across the sky does not.

The margin matters and is the whole reason this is not a naive point-in-polygon test. A lane *boundary* runs
along the edge of the drivable region and its points legitimately fall a little outside it, so demanding
strict containment would reject exactly the road-edge lanes the ontology cares most about. Points are
allowed to sit within a small fraction of the frame width of the surface.

Measured before it was written, against 1,500 real lanes on frames that carry a drivable mask: 98.7% have
more than 60% of their points on or near the surface, and 1.3% fall below. A filter should cut a tail, not a
population, and that is the shape of one that does.

A geometric test was tried first and abandoned: reject a lane wider than it is tall, on the reasoning that a
lane in a forward camera runs toward the vanishing point. On this corpus 73.6% of lanes are wider than tall,
because a lane near the horizon really is nearly horizontal. It would have purged three quarters of the
corpus. The measurement is the only reason that is not in here.
"""

from __future__ import annotations

import numpy as np

from core.logging import get_logger

log = get_logger("lane_plausible")

# How far off the drivable surface a control point may sit, as a fraction of frame width. A lane boundary
# runs along the road's edge, so some slack is the difference between a filter and a rule that deletes every
# road edge.
SURFACE_MARGIN_FRAC = 0.03
# How much of a lane has to be on the surface. Below this the line is somewhere else in the scene: a wall, a
# roofline, a bridge. Set where the corpus separates: real lanes cluster at 1.0, the tail sits at 0.0.
MIN_ON_SURFACE = 0.5
# Classes whose polygons count as road. Non-drivable is deliberately excluded: it is the pavement and the
# median, and a lane line does not run along those.
SURFACE_CLASSES = ("drivable", "fallback")


def on_surface_fraction(control_points: list, surface_polygons: list, margin_px: float) -> float | None:
    """How much of the lane lies on, or just off, the drivable surface.

    Returns None when there is nothing to compare against, which is not the same as zero and must not be
    treated as one: a frame with no drivable mask is a frame nobody has segmented, and refusing its lanes
    would make the proposer depend on a second model having run first.
    """
    import cv2

    rings = []
    for flat in (surface_polygons or []):
        pts = np.asarray(flat, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        if len(pts) >= 3:
            rings.append(pts)
    if not rings:
        return None

    lane = [[float(p[0]), float(p[1])] for p in (control_points or [])
            if p is not None and len(p) >= 2]
    if not lane:
        return None

    # Signed distance to the nearest surface: positive inside, negative outside. The best over all rings,
    # because the road is often several disjoint polygons either side of a traffic island.
    best = np.full(len(lane), -1e9, dtype=np.float64)
    for ring in rings:
        d = np.array([cv2.pointPolygonTest(ring, (float(x), float(y)), True) for x, y in lane])
        best = np.maximum(best, d)
    return float((best >= -margin_px).mean())


def surface_polygons(drivable_classes: dict | None) -> list:
    """The road polygons out of a drivable mask payload."""
    if not drivable_classes:
        return []
    out: list = []
    for name in SURFACE_CLASSES:
        out.extend(drivable_classes.get(name) or [])
    return out


def is_plausible(control_points: list, drivable_classes: dict | None, frame_width: int,
                 *, min_on_surface: float = MIN_ON_SURFACE) -> tuple[bool, dict]:
    """Whether this proposal is a lane, and the evidence either way.

    Returns (plausible, evidence). Plausible when there is no surface to check against, deliberately: the
    absence of a drivable mask is missing evidence rather than evidence of absence, and a proposer that
    silently produced nothing until somebody had run the segmenter would be a worse failure than the one
    this fixes.
    """
    polys = surface_polygons(drivable_classes)
    margin = SURFACE_MARGIN_FRAC * max(1, int(frame_width or 0))
    frac = on_surface_fraction(control_points, polys, margin)
    if frac is None:
        return True, {"checked": False,
                      "reason": "no drivable surface on this frame to compare against"}
    ok = frac >= min_on_surface
    return ok, {
        "checked": True,
        "on_surface": round(frac, 3),
        "min_required": min_on_surface,
        "margin_px": round(margin, 1),
        "reason": ("on the road surface" if ok else
                   "the line does not lie on the road: it is somewhere else in the scene, which is what "
                   "a hoarding, a flyover girder or a kerb looks like to a lane detector"),
    }


def filter_proposals(proposals: list, drivable_classes: dict | None, frame_width: int) -> tuple[list, list]:
    """Split proposals into the ones on the road and the ones that are not.

    Both halves come back. The caller logs or reports what it dropped rather than discarding it quietly: a
    proposer that silently halves its own output is one nobody can debug.
    """
    kept, rejected = [], []
    for cps in (proposals or []):
        ok, ev = is_plausible(cps, drivable_classes, frame_width)
        (kept if ok else rejected).append((cps, ev))
    return kept, rejected
