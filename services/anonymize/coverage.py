"""Whether a blur actually covered anything, as opposed to whether a frame was looked at.

The DPDPA export gate has always checked that every exported frame carries a PiiAudit row. That is a
fail-closed check against *never having run the anonymizer*, and it is not the same question as whether the
run found what was there. `services/anonymize/recheck.py` measured the difference on this corpus: 82.4% of
frames holding an annotated person have zero faces redacted, and every one of those frames has an audit row.
The gate passed all of them.

The predicate here is annotation-relative, and it has to be. `recheck.py` deliberately writes a zero-count
audit row when it looks and finds nothing, so that a checked frame is not mistaken for an unchecked one - a
gate keyed on `n_faces > 0` would refuse every legitimately empty highway frame in the corpus. Instead: the
frame's own annotations say where the people and vehicles are, so a frame is only suspect when it carries an
annotation that implies a redaction and the audit holds no region covering it.

Split out of recheck.py rather than added to it because this runs inside the export request path, and
recheck.py imports cv2, numpy and the recall backends at module scope. The shared constants live here and
recheck.py imports them back, so the window this measures coverage in is by construction the same window the
re-check would look in - a drift between them would mean refusing a frame the remediation cannot fix.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

# Which annotated classes imply which redaction, by ontology group rather than by name, so a class added
# later inherits the behaviour instead of falling outside a hardcoded list. Faces include the two- and
# three-wheeler groups because a rider's head routinely sits inside the vehicle's box rather than their own.
FACE_GROUPS: frozenset[str] = frozenset({"vru", "two_wheeler", "three_wheeler"})
PLATE_GROUPS: frozenset[str] = frozenset({"two_wheeler", "three_wheeler", "four_wheeler", "heavy"})
GROUPS_BY_TARGET: dict[str, frozenset[str]] = {"face": FACE_GROUPS, "plate": PLATE_GROUPS}

# The l1 groups worth fetching at all. Pushed into SQL so a frame carrying only infrastructure annotations
# produces no rows and never reaches Python.
REDACTION_L1: frozenset[str] = FACE_GROUPS | PLATE_GROUPS

# A face sits at the top of a person's box and a plate straddles the edge of a tightly drawn vehicle box, so
# the window is margined. Must stay identical to recheck._MARGIN, which imports these.
MARGIN = 0.15
MIN_CROP_PX = 16

# The fraction of a stored region's OWN area that must fall inside the annotation window for that region to
# count as covering it.
#
# This is containment, not IoU, and the distinction is the whole design. A real 40x40 face inside a 100x300
# person box has an IoU of about 0.05 against that box - nowhere near any usable threshold - so an
# IoU-based rule would report every frame in the corpus as uncovered and refuse the entire export.
# recheck._COVERED_IOU is 0.3 and is the right metric for what it does, which is comparing a candidate face
# against a stored face at the same scale; it is the wrong metric here and is deliberately not reused.
COVERED_FRACTION = 0.5


@dataclass(frozen=True)
class Demand:
    """One annotation obliging a redaction of `target` somewhere inside `window` (full-frame pixels)."""

    target: str                              # "face" | "plate"
    window: tuple[int, int, int, int]
    class_name: str


def crop_window(box: Sequence[float], w: int, h: int) -> tuple[int, int, int, int] | None:
    """A margined, frame-clamped window around one annotation, or None if too small to be worth reading."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
    cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    cx2, cy2 = min(w, int(round(x2 + mx))), min(h, int(round(y2 + my)))
    if cx2 - cx1 < MIN_CROP_PX or cy2 - cy1 < MIN_CROP_PX:
        return None
    return cx1, cy1, cx2, cy2


def demands(rows: Iterable[tuple[Sequence[float], str, str, str]], w: int, h: int,
            targets: Iterable[str] = ("face", "plate")) -> list[Demand]:
    """What a frame's annotations oblige. `rows` are (bbox, l0, l1, class_name), as stored.

    Mirrors recheck.roi_windows' selection exactly, minus the database: same groups, same window, same
    minimum size. An annotation too small to crop raises no demand, because the re-check could not look
    inside it either and a demand nothing can satisfy is a refusal with no route out.
    """
    wanted = tuple(targets)
    out: list[Demand] = []
    for bbox, l0, l1, class_name in rows:
        if l0 != "object" or not bbox:
            continue
        window = crop_window(bbox, w, h)
        if window is None:
            continue
        for target in wanted:
            if l1 in GROUPS_BY_TARGET.get(target, frozenset()):
                out.append(Demand(target=target, window=window, class_name=class_name))
    return out


def inside_fraction(region_bbox: Sequence[float], window: tuple[int, int, int, int]) -> float:
    """The fraction of the REGION's own area that lies inside the window.

    Deliberately asymmetric. A face is a small part of the person carrying it, so the question is "is this
    blur on this person", not "do these two rectangles look alike".
    """
    try:
        rx1, ry1, rx2, ry2 = (float(v) for v in region_bbox[:4])
    except (TypeError, ValueError):
        return 0.0
    area = (rx2 - rx1) * (ry2 - ry1)
    if area <= 0:
        return 0.0
    wx1, wy1, wx2, wy2 = window
    ix = max(0.0, min(rx2, wx2) - max(rx1, wx1))
    iy = max(0.0, min(ry2, wy2) - max(ry1, wy1))
    return (ix * iy) / area


def unmet(demands_: Sequence[Demand], regions: Iterable[dict[str, Any]]) -> list[Demand]:
    """The demands no stored region satisfies.

    Matching is greedy and one-to-one: a single blurred face on a frame carrying three annotated pedestrians
    covers one of them, not all three. A region satisfies at most one demand, and each demand takes the
    region that sits most fully inside it, so the leftovers are the people nothing was blurred for.
    """
    usable: list[tuple[str, list[float]]] = []
    for r in regions or []:
        if not isinstance(r, dict):
            continue
        bbox = r.get("bbox")
        # A region with no usable bbox is not evidence of anything. It cannot be shown to cover a person,
        # and this gate exists precisely because a row's existence was being taken as proof.
        if not bbox or len(bbox) < 4:
            continue
        usable.append((str(r.get("type", "")), list(bbox)))

    taken: set[int] = set()
    missing: list[Demand] = []
    for d in demands_:
        best_i, best_frac = None, 0.0
        for i, (rtype, bbox) in enumerate(usable):
            if i in taken or rtype != d.target:
                continue
            frac = inside_fraction(bbox, d.window)
            if frac > best_frac:
                best_i, best_frac = i, frac
        if best_i is not None and best_frac >= COVERED_FRACTION:
            taken.add(best_i)
        else:
            missing.append(d)
    return missing


def summarize(missing: Sequence[Demand]) -> dict[str, Any]:
    """The per-frame payload shape the gate reports: what is missing, and for which classes."""
    counts: dict[str, int] = {}
    classes: list[str] = []
    for d in missing:
        counts[d.target] = counts.get(d.target, 0) + 1
        if d.class_name not in classes:
            classes.append(d.class_name)
    return {"missing": counts, "classes": sorted(classes)}
