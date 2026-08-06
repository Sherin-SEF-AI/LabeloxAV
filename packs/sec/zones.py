"""Zones and line crossings on a fixed camera, and the incidents they raise.

A static camera's whole value is that its frame does not move. That makes a polygon drawn on it a permanent
statement about the world: this quadrilateral is the loading bay, this segment is the gate line. The Sec
pack had the static-camera scene model and no way to say either, so every detection was a detection
somewhere in the picture and no rule could be written about place.

The geometry is deliberately plain, because the mistakes here are not in the algorithms:

- **Point-in-polygon by ray casting**, on the box's bottom-centre rather than its centroid. An object stands
  on the ground at the bottom of its box; using the centroid puts a tall person in the zone while their feet
  are still outside it, which is exactly wrong for a floor plan.
- **Crossing by side change**, comparing which side of the line the track was on between two frames. Testing
  intersection with the box instead would fire continuously for as long as a wide object straddled the line.
- **Dwell measured from the first frame inside**, and reported once. A dwell rule that re-fires every frame
  produces one incident per frame of loitering, which is the same as having no rule.
"""

from __future__ import annotations

from core.logging import get_logger
from packs.base import Crossing

log = get_logger("sec_zones")

RULES = ("enter", "exit", "dwell", "cross")
KINDS = ("area", "line")

# Re-exported so this module stays the one place a reader looks for zone vocabulary, while the shape itself
# lives in the contract because the engine reads every field of it.
__all__ = ["Crossing", "KINDS", "RULES", "anchor_point", "evaluate_track", "point_in_polygon",
           "segment_span", "side_of_line", "validate_zone"]


def anchor_point(bbox: list[float]) -> tuple[float, float]:
    """Where an object touches the ground: the bottom centre of its box.

    Not the centroid. An object stands at the bottom of its box, so a centroid test puts a tall person
    inside a floor zone while their feet are still outside it, and takes them out again while they are
    still standing in it.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    return ((x1 + x2) / 2.0, max(y1, y2))


def point_in_polygon(pt: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Ray casting. Returns False for a degenerate polygon rather than raising."""
    if not polygon or len(polygon) < 3:
        return False
    x, y = pt
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = float(polygon[i][0]), float(polygon[i][1])
        x2, y2 = float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])
        # The half-open y test is what stops a vertex exactly on the ray being counted twice.
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1) if y2 != y1 else 0.0
            if x < x1 + t * (x2 - x1):
                inside = not inside
    return inside


def side_of_line(pt: tuple[float, float], line: list[list[float]]) -> int:
    """Which side of a directed segment a point is on: 1, -1, or 0 exactly on it."""
    if not line or len(line) < 2:
        return 0
    (ax, ay), (bx, by) = (float(line[0][0]), float(line[0][1])), (float(line[1][0]), float(line[1][1]))
    x, y = pt
    cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
    if abs(cross) < 1e-9:
        return 0
    return 1 if cross > 0 else -1


def segment_span(pt_a: tuple[float, float], pt_b: tuple[float, float],
                 line: list[list[float]]) -> bool:
    """Whether the movement from a to b actually passes within the line's extent.

    A side change alone is not a crossing: a segment is finite, and an object moving from one side of the
    infinite line to the other, far off the end of the drawn segment, has not crossed the gate.
    """
    if not line or len(line) < 2:
        return False
    (ax, ay), (bx, by) = (float(line[0][0]), float(line[0][1])), (float(line[1][0]), float(line[1][1]))
    (px, py), (qx, qy) = pt_a, pt_b
    d1 = (qx - px, qy - py)
    d2 = (bx - ax, by - ay)
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return False   # parallel
    t = ((ax - px) * d2[1] - (ay - py) * d2[0]) / denom
    u = ((ax - px) * d1[1] - (ay - py) * d1[0]) / denom
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def evaluate_track(zone: dict, samples: list[dict]) -> list[Crossing]:
    """Run one zone's rule over one track's ordered samples.

    A track rather than a frame, because every rule here is about change: entering, leaving, crossing and
    dwelling are all statements about two moments, and a per-frame evaluation cannot express any of them.
    """
    rule = zone.get("rule", "enter")
    if rule not in RULES or not samples:
        return []
    classes = set(zone.get("classes") or [])
    samples = sorted(samples, key=lambda s: int(s["ts_ns"]))
    if classes:
        samples = [s for s in samples if s.get("class_name") in classes]
        if not samples:
            return []

    points = zone.get("points") or []
    out: list[Crossing] = []
    track_id = samples[0].get("track_id")
    class_name = samples[0].get("class_name", "object")

    def _fire(rule_name: str, ts: int, detail: dict) -> None:
        out.append(Crossing(zone_id=str(zone["zone_id"]), zone_name=zone.get("name", ""),
                            rule=rule_name, track_id=str(track_id) if track_id else None,
                            class_name=class_name, ts_ns=int(ts),
                            severity=zone.get("severity", "warn"), detail=detail))

    if zone.get("kind") == "line" or rule == "cross":
        prev = None
        for s in samples:
            pt = anchor_point(s["bbox"])
            side = side_of_line(pt, points)
            if prev is not None and prev[1] != 0 and side != 0 and side != prev[1] \
                    and segment_span(prev[0], pt, points):
                _fire("cross", s["ts_ns"],
                      {"from_side": prev[1], "to_side": side,
                       # Direction is carried because "in" and "out" of a gate are different events, and a
                       # rule that cannot tell them apart cannot express either.
                       "direction": "a_to_b" if side > 0 else "b_to_a"})
            prev = (pt, side)
        return out

    inside_since: int | None = None
    was_inside = False
    dwell_fired = False
    for s in samples:
        inside = point_in_polygon(anchor_point(s["bbox"]), points)
        if inside and not was_inside:
            inside_since = int(s["ts_ns"])
            dwell_fired = False
            if rule == "enter":
                _fire("enter", s["ts_ns"], {})
        elif not inside and was_inside:
            if rule == "exit":
                _fire("exit", s["ts_ns"], {"dwelled_s": _seconds(inside_since, s["ts_ns"])})
            inside_since = None
        elif inside and rule == "dwell" and not dwell_fired and inside_since is not None:
            threshold = float(zone.get("dwell_seconds") or 30.0)
            held = _seconds(inside_since, s["ts_ns"])
            if held >= threshold:
                # Once per visit, not once per frame. A dwell rule that re-fires produces one incident per
                # frame of loitering, which is the same as having no rule at all.
                dwell_fired = True
                _fire("dwell", s["ts_ns"], {"dwelled_s": round(held, 2), "threshold_s": threshold})
        was_inside = inside
    return out


def _seconds(a: int | None, b: int) -> float:
    if a is None:
        return 0.0
    return max(0.0, (int(b) - int(a)) / 1e9)


def validate_zone(kind: str, rule: str, points: list, dwell_seconds: float | None) -> None:
    """Refuse a zone that cannot mean anything, at creation rather than at evaluation."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    pts = points or []
    if kind == "line":
        if len(pts) != 2:
            raise ValueError("a line zone needs exactly two points")
        if rule != "cross":
            raise ValueError("a line zone only supports the 'cross' rule")
    else:
        if len(pts) < 3:
            raise ValueError("an area zone needs at least three points")
        if rule == "cross":
            raise ValueError("'cross' is a line rule; use enter, exit or dwell on an area")
    if rule == "dwell" and not dwell_seconds:
        raise ValueError("a dwell rule needs dwell_seconds; without one it fires on entry")
    if any(len(p) < 2 for p in pts):
        raise ValueError("every point needs an x and a y")
