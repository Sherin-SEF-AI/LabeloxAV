"""The self-consistency critic. The single-frame quality reviewer (services/autolabel/quality_reviewer.py)
already catches sky-boxes, impossible sizes, and part-of-vehicle nonsense within one image. This critic adds
the checks that need CONTEXT the single frame does not have:

- temporal: a tracked object should not flip class (car->truck->car) or teleport between consecutive frames.
- geometric: a thing that must sit on the road (a vehicle, a pedestrian) whose box bottom is above the
  horizon has no ground point -- it cannot physically be there.
- motion: a pedestrian doing 60 km/h, or anything doing 250 km/h, is a bad label or a broken track.
- cross-modal: a solid vehicle box with essentially no LiDAR returns inside it disagrees with the sensor.
- relationship: a "rider" with no two-wheeler under/beside it is usually a misread pedestrian.

Every check is a VETO only: it can demote an auto-accept to human review, never create one. Each check
no-ops cleanly when its input is missing (no LiDAR, no dynamics, no track), so it is safe on plain dashcam
frames where only the temporal/geometric/relationship checks have data. It stays conservative on purpose:
a false flag costs one human glance, a missed error would auto-accept a wrong label.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Class-name groups the checks reason about. Kept as names (not ids) so they survive ontology renumbering.
_TWO_WHEELERS = {"motorcycle", "scooter", "moped", "cycle", "bicycle", "delivery_rider_bike", "autorickshaw"}
_VRU = {"pedestrian", "rider", "cyclist", "motorcyclist", "bicyclist"}
_ANIMALS = {"cattle", "dog", "cow", "buffalo", "goat", "animal_fallback"}
# Things that must rest on the road (so the horizon check applies). Elevated things (signs, lights,
# hoardings, traffic signals) are intentionally excluded -- they are legitimately above the horizon.
_GROUND_L1 = {"vehicle"}
_MAX_VRU_KMH = 45.0        # a person/cyclist above this is almost certainly a bad label or track
_MAX_ANY_KMH = 220.0       # nothing on an Indian road plausibly exceeds this; a track glitch if it does
_TELEPORT_FRAC = 0.6       # centroid jump > 60% of the frame diagonal between consecutive frames = teleport
_MIN_LIDAR_PTS = 3         # a close, large vehicle box with fewer returns than this disagrees with the sensor


@dataclass
class CriticVerdict:
    ok: bool = True
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, str] = field(default_factory=dict)  # check_name -> "pass" | "flag" | "skip"

    def flag(self, check: str, reason: str) -> None:
        self.ok = False
        self.reasons.append(reason)
        self.checks[check] = "flag"

    def passed(self, check: str) -> None:
        self.checks.setdefault(check, "pass")

    def skipped(self, check: str) -> None:
        self.checks.setdefault(check, "skip")


@dataclass
class CriticContext:
    """Everything the critic needs, pre-loaded by the frame agent so the checks stay pure and fast."""
    onto: object                                   # Ontology (by_id/by_name)
    cam_id: str
    width: int
    height: int
    frame_objects: list                            # the frame's objects (ORM rows) for cross-object checks
    dynamics: dict = field(default_factory=dict)   # object_id(str) -> {speed_kmh, ...}
    track_history: dict = field(default_factory=dict)  # track_id(str) -> [(ts_ns, class_id, cx, cy), ...]
    cloud_xyz: object | None = None                # np.ndarray (N,3) ego points, or None if no LiDAR


def _name(onto, class_id: int) -> str:
    try:
        return onto.by_id(int(class_id)).name
    except Exception:  # noqa: BLE001
        return ""


def _l1(onto, class_id: int) -> str:
    try:
        return onto.by_id(int(class_id)).l1
    except Exception:  # noqa: BLE001
        return ""


def _centroid(bbox) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _check_temporal(obj, v: CriticVerdict, c: CriticContext) -> None:
    tid = getattr(obj, "track_id", None)
    hist = c.track_history.get(str(tid)) if tid else None
    if not hist or len(hist) < 2:
        v.skipped("temporal")
        return
    classes = {cid for (_ts, cid, _cx, _cy) in hist}
    if len(classes) > 1:
        names = sorted({_name(c.onto, cid) or str(cid) for cid in classes})
        v.flag("temporal", f"track class flips across frames: {', '.join(names)}")
    # teleport: consecutive centroids jumping most of the frame in one step
    diag = (c.width ** 2 + c.height ** 2) ** 0.5
    ordered = sorted(hist, key=lambda r: r[0])
    for (t0, _c0, x0, y0), (t1, _c1, x1, y1) in zip(ordered, ordered[1:]):
        if diag > 0 and ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 > _TELEPORT_FRAC * diag:
            v.flag("temporal", "track centroid teleports between consecutive frames")
            break
    v.passed("temporal")


def _check_geometric(obj, v: CriticVerdict, c: CriticContext) -> None:
    name = _name(c.onto, obj.class_id)
    if _l1(c.onto, obj.class_id) not in _GROUND_L1 and name not in _VRU and name not in _ANIMALS:
        v.skipped("geometric")
        return
    try:
        from services.lidar.project import camera_ray_to_ego
        cx, _cy = _centroid(obj.bbox)
        v_bottom = float(obj.bbox[3])  # box bottom edge is where a ground object contacts the road
        ray = camera_ray_to_ego(cx, v_bottom, c.cam_id, c.width, c.height)
        dz = float(ray["direction"][2])
        oz = float(ray["origin"][2])
        t = (-oz / dz) if abs(dz) > 1e-6 else -1.0
        if t <= 0:
            v.flag("geometric", f"{name or 'ground object'} bottom is above the horizon (no ground point)")
        else:
            v.passed("geometric")
    except Exception:  # noqa: BLE001 -- no calibration for this cam, etc.: cannot judge, so do not veto
        v.skipped("geometric")


def _check_motion(obj, v: CriticVerdict, c: CriticContext) -> None:
    dyn = c.dynamics.get(str(obj.object_id))
    spd = dyn.get("speed_kmh") if dyn else None
    if spd is None:
        v.skipped("motion")
        return
    name = _name(c.onto, obj.class_id)
    if name in _VRU and spd > _MAX_VRU_KMH:
        v.flag("motion", f"{name} moving {spd:.0f} km/h (implausible for a vulnerable road user)")
    elif spd > _MAX_ANY_KMH:
        v.flag("motion", f"{name or 'object'} moving {spd:.0f} km/h (implausible)")
    else:
        v.passed("motion")


def _check_cross_modal(obj, v: CriticVerdict, c: CriticContext) -> None:
    if c.cloud_xyz is None or _l1(c.onto, obj.class_id) not in _GROUND_L1:
        v.skipped("cross_modal")
        return
    try:
        from services.lidar.detect3d.lift import frustum_indices
        idx = frustum_indices(c.cloud_xyz, list(obj.bbox), c.cam_id, c.width, c.height)
        n = int(len(idx))
        # Only veto egregiously empty boxes for reasonably large vehicles (small/distant boxes legitimately
        # catch few returns, so requiring points there would over-flag).
        area = max(0.0, float(obj.bbox[2]) - float(obj.bbox[0])) * max(0.0, float(obj.bbox[3]) - float(obj.bbox[1]))
        big = area > 0.02 * c.width * c.height
        if big and n < _MIN_LIDAR_PTS:
            v.flag("cross_modal", f"vehicle box has {n} LiDAR returns (sensor sees nothing there)")
        else:
            v.passed("cross_modal")
    except Exception:  # noqa: BLE001
        v.skipped("cross_modal")


def _check_relationship(obj, v: CriticVerdict, c: CriticContext) -> None:
    name = _name(c.onto, obj.class_id)
    if name != "rider":
        v.skipped("relationship")
        return
    # a rider should sit on/over a two-wheeler: look for one whose box overlaps this one's lower half
    ox1, oy1, ox2, oy2 = (float(x) for x in obj.bbox)
    lower_y = oy1 + 0.5 * (oy2 - oy1)
    for other in c.frame_objects:
        if other.object_id == obj.object_id:
            continue
        if _name(c.onto, other.class_id) not in _TWO_WHEELERS:
            continue
        bx1, by1, bx2, by2 = (float(x) for x in other.bbox)
        if bx1 < ox2 and bx2 > ox1 and by2 > lower_y and by1 < oy2:  # horizontal overlap + reaches lower half
            v.passed("relationship")
            return
    v.flag("relationship", "rider with no two-wheeler beneath it (likely a pedestrian)")


def critique_frame(ctx: CriticContext) -> dict[str, CriticVerdict]:
    """Run every applicable check on every object in the frame; return object_id(str) -> verdict."""
    out: dict[str, CriticVerdict] = {}
    for obj in ctx.frame_objects:
        v = CriticVerdict()
        _check_temporal(obj, v, ctx)
        _check_geometric(obj, v, ctx)
        _check_motion(obj, v, ctx)
        _check_cross_modal(obj, v, ctx)
        _check_relationship(obj, v, ctx)
        out[str(obj.object_id)] = v
    return out
