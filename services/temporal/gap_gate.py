"""Whether two detections on a track are plausibly the same object, and therefore whether the hole between
them may be filled.

This is the gate that was missing, and its absence is the whole defect. Interpolation arithmetic is not what
went wrong in the 137,913-object gap fill: printing a track's boxes in time order shows the fills bridging
two real detections smoothly and landing where they should. What went wrong is that most of those pairs were
not the same object. 45.9% of track steps have zero box overlap and 59.2% of tracks contain a centre jump of
more than a quarter of the frame width, because the tracker's feasibility test admits a match on appearance
alone at zero overlap. Interpolating across such a pair draws a smooth path between two unrelated things and
puts every box on empty road.

The gate is worth its own module because the same predicate answers two questions: may this hole be filled,
and should this track be split here. One definition, so the splitter and the filler cannot disagree about
what a discontinuity is.

Measured against the gaps that were actually filled: 64.6% of endpoint pairs stayed within the displacement
bound, 70.3% within the scale bound, 34.4% agreed on class, and 20.9% passed all three. Judged precision of
the objects produced was 0.209. The correspondence is the evidence that this is the right cut.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.logging import get_logger

log = get_logger("temporal.gap_gate")

# Fraction of frame width a centre may travel between two anchors before they stop being one object.
#
# Generous on purpose. Ingest runs at 3 fps (`IngestSettings.target_fps`), so consecutive frames are ~333 ms
# apart and smart extract widens that further; a nearby vehicle crossing the field of view legitimately moves
# a long way between samples. The bound is here to reject teleports, not to enforce smooth motion.
MAX_CENTRE_TRAVEL_FRAC = 0.25

# Largest area ratio between two anchors.
#
# Measured against 118,881 consecutive detection pairs that already pass the displacement bound - so pairs
# very likely to be one object - the ratio distribution is p50 1.24, p90 3.90, p95 6.93, p99 24.21. Cutting
# at 4.0 sits on the 90th percentile and keeps 90.3% of them.
#
# That is a deliberate precision trade and it is the right way round here: this gate exists because 109,000
# objects were created between things that were not the same object, so refusing a tenth of the genuine
# growth cases costs recall that a later pass can recover, while admitting them costs corpus damage that
# nothing recovers. Raise it toward 7 to sit at p95 if recall matters more than it does today.
MAX_AREA_RATIO = 4.0

# Longest hole that may be filled at all, in frames. At 3 fps this is four seconds of unobserved motion.
MAX_GAP_FRAMES = 12


@dataclass(frozen=True)
class GateResult:
    """Why a hole was or was not filled. `reason` is None when it passed."""

    ok: bool
    reason: str | None = None
    # What the gate measured, so a rejection can be argued with rather than only obeyed.
    detail: dict | None = None


def _centre(box) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def _area(box) -> float:
    return max(1.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])))


def same_object(box_a, box_b, class_a: str, class_b: str, *, frame_width: float,
                gap_frames: int, cliques=None,
                max_travel_frac: float = MAX_CENTRE_TRAVEL_FRAC,
                max_area_ratio: float = MAX_AREA_RATIO,
                max_gap_frames: int = MAX_GAP_FRAMES) -> GateResult:
    """Could these two detections be the same object, such that the frames between them may be filled?

    Deliberately three cheap geometric and semantic tests rather than an appearance model. Appearance is what
    the tracker already over-trusted: its gate admits a match when the DINOv3 cosine clears 0.55, which two
    arbitrary same-class vehicles routinely do. Geometry is the signal appearance was allowed to override.

    `cliques` is the pack's `CliqueSpec`. Class equality is deliberately not required: the detector renames
    one object between consecutive frames - a single receding vehicle in this corpus is labelled truck,
    rider, autorickshaw, suv and motorcycle on five consecutive frames - so exact matching would reject
    65.6% of holes, most of them genuine. A shared confusion clique tolerates that instability without
    letting a pedestrian bridge to a bus.
    """
    if gap_frames > max_gap_frames:
        return GateResult(False, "gap_too_long", {"gap_frames": gap_frames, "max": max_gap_frames})

    ax, ay = _centre(box_a)
    bx, by = _centre(box_b)
    travel = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    limit = max(1.0, float(frame_width) * max_travel_frac)
    if travel > limit:
        return GateResult(False, "endpoints_teleport",
                          {"travel_px": round(travel, 1), "limit_px": round(limit, 1)})

    aa, ab = _area(box_a), _area(box_b)
    ratio = max(aa, ab) / min(aa, ab)
    if ratio > max_area_ratio:
        return GateResult(False, "endpoints_scale_jump",
                          {"area_ratio": round(ratio, 2), "max": max_area_ratio})

    if class_a != class_b:
        # No clique spec means the domain declares no confusions, so the only safe reading of two different
        # class names is two different objects.
        ca = cliques.clique_of(class_a) if cliques is not None else None
        cb = cliques.clique_of(class_b) if cliques is not None else None
        if ca is None or ca is not cb:
            return GateResult(False, "endpoints_class_mismatch",
                              {"class_a": class_a, "class_b": class_b,
                               "clique_a": ca.name if ca else None, "clique_b": cb.name if cb else None})

    return GateResult(True, None, {"travel_px": round(travel, 1), "area_ratio": round(ratio, 2),
                                   "gap_frames": gap_frames})


def is_discontinuity(box_a, box_b, *, frame_width: float,
                     max_travel_frac: float = MAX_CENTRE_TRAVEL_FRAC) -> bool:
    """True when a track step cannot be one object continuing, and the track should be cut here.

    The geometric half of `same_object`, without the class test: a track that changes class is the
    detector being unstable, while a track that teleports is two objects wearing one id. Splitting on class
    alone would shred correct tracks.
    """
    ax, ay = _centre(box_a)
    bx, by = _centre(box_b)
    travel = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    if travel > max(1.0, float(frame_width) * max_travel_frac):
        return True
    # Zero overlap on its own is not enough at 3 fps - a small fast object legitimately clears its own box
    # between samples - so it counts only alongside a large area change, which together read as two objects.
    ix = max(0.0, min(float(box_a[2]), float(box_b[2])) - max(float(box_a[0]), float(box_b[0])))
    iy = max(0.0, min(float(box_a[3]), float(box_b[3])) - max(float(box_a[1]), float(box_b[1])))
    if ix * iy > 0:
        return False
    aa, ab = _area(box_a), _area(box_b)
    return max(aa, ab) / min(aa, ab) > MAX_AREA_RATIO
