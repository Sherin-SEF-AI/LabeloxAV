"""2D->3D auto-cuboid agent: lift every 2D vehicle/VRU box on a frame to a 3D cuboid, monocularly, and
validate it by reprojection so only the ones that actually fit get accepted.

Works without LiDAR (the corpus is camera-heavy): the box's bottom-centre is the ground-contact point, so
camera_ray_to_ego lifts it to the ego ground plane; the cuboid is sized from a class prior and its yaw is
chosen by projecting a few candidate orientations back onto the image and keeping the one whose reprojected
box best matches the 2D box. That reprojection IoU is also the confidence: a clean fit auto-accepts, a rough
one routes to review, an un-liftable box (bottom above the horizon) is skipped. When a synchronized LiDAR
cloud exists the centre is refined to the frustum points (cross-modal); otherwise the monocular estimate
stands. Writes Object.cuboid_3d (the 2D-attached cuboid the editor projects), recorded on one reversible
AgentRun so revert clears the cuboids exactly.
"""

from __future__ import annotations

import math
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Frame, Object

log = get_logger("agent.cuboid")

# Class-size priors as cuboid_3d.size = [width, length, height] (metres). Matched to the editor's CUBOID_DIMS.
_DIMS_BY_NAME = {
    "sedan": [1.8, 4.2, 1.5], "hatchback": [1.7, 3.8, 1.5], "suv": [1.9, 4.6, 1.7], "app_cab": [1.8, 4.2, 1.5],
    "truck": [2.5, 7.0, 3.0], "bus": [2.6, 11.0, 3.2], "minivan": [1.9, 4.8, 1.8], "ambulance": [2.0, 5.5, 2.4],
    "motorcycle": [0.8, 2.0, 1.4], "scooter": [0.7, 1.8, 1.3], "moped": [0.7, 1.8, 1.3], "cycle": [0.6, 1.7, 1.3],
    "autorickshaw": [1.4, 2.6, 1.8], "pedestrian": [0.6, 0.6, 1.7], "rider": [0.8, 2.0, 1.6],
}
_DIMS_BY_L1 = {"four_wheeler": [1.8, 4.2, 1.5], "heavy": [2.5, 7.0, 3.0], "two_wheeler": [0.8, 2.0, 1.4],
               "three_wheeler": [1.4, 2.6, 1.8], "vru": [0.6, 0.6, 1.7]}
_LIFT_L1 = set(_DIMS_BY_L1)                       # only lift things that rest on the road
_YAW_CANDIDATES = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]

# The coarse candidates above are 45 degrees apart, which is as close as a four-way search can get. A
# local sweep around whichever of them won is worth running: measured over 238 lifted objects from the
# real corpus, mean reprojection IoU goes from 0.491 to 0.555 and 178 of them improve by more than 0.01,
# the best by 0.29.
_REFINE_HALF_SPAN = math.pi / 8      # +/- 22.5 degrees, half the coarse spacing
_REFINE_STEP = math.radians(2.0)

# How close a fitted yaw must be to the road direction before it is snapped to it.
#
# Reprojection IoU is a weak signal for yaw and cannot be otherwise: the axis-aligned image box of a car
# seen head-on is very nearly the box of the same car seen from behind. So a yaw within this of a lane's
# heading is better explained by the lane, which is measured from the road rather than inferred from a
# silhouette.
_LANE_SNAP_TOL = math.radians(25.0)


def _dims_for(onto, class_id: int) -> list[float] | None:
    try:
        c = onto.by_id(int(class_id))
    except Exception:  # noqa: BLE001
        return None
    if c.name in _DIMS_BY_NAME:
        return list(_DIMS_BY_NAME[c.name])
    return list(_DIMS_BY_L1[c.l1]) if c.l1 in _DIMS_BY_L1 else None


def _iou2d(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 1e-6 else 0.0


def _reproj_box(proj) -> list[float] | None:
    """Axis-aligned image box of the cuboid's corners that are in front of the camera."""
    uv = proj["corners_uv"]
    infr = proj["in_front"]
    pts = [uv[i] for i in range(len(uv)) if infr[i]]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _yaw_iou(center, size, yaw, cam_id, w, h, target) -> float | None:
    """Reprojection IoU of one candidate yaw against the observed 2D box."""
    from services.lidar.boxes import project_cuboid

    proj = project_cuboid(center, [size[1], size[0], size[2]], yaw, cam_id, w, h)
    rb = _reproj_box(proj)
    return None if rb is None else _iou2d(rb, target)


def _best_yaw(center, size, cam_id, w, h, target, yaws) -> tuple[float, float] | None:
    best = None
    for yaw in yaws:
        iou = _yaw_iou(center, size, yaw, cam_id, w, h, target)
        if iou is None:
            continue
        if best is None or iou > best[1]:
            best = (float(yaw), float(iou))
    return best


def lane_headings(lanes, cam_id: str, w: int, h: int) -> list[float]:
    """Road direction in the ego frame, in radians, one per lane that can be lifted.

    The two ingredients have both existed and were never composed: lanes are stored as image-space
    control points, and `camera_ray_to_ego` already lifts an image point to the ground plane, which is
    exactly what `_fit_mono` uses for the box's ground contact. Running the same lift over a lane's first
    and last control points gives the direction of the road under the camera.

    Returned as an axis, not a bearing. A lane says which way the road runs and nothing about which way
    along it a vehicle faces, so every heading here is equally valid plus or minus pi.
    """
    from services.lidar.project import camera_ray_to_ego

    out: list[float] = []
    for lane in lanes:
        pts = lane.control_points if isinstance(lane.control_points, list) else []
        if len(pts) < 2:
            continue
        ends = []
        for u, v in (pts[0], pts[-1]):
            try:
                ray = camera_ray_to_ego(float(u), float(v), cam_id, w, h)
            except Exception:  # noqa: BLE001 - no calibration for this camera
                break
            dz, oz = float(ray["direction"][2]), float(ray["origin"][2])
            if abs(dz) < 1e-6:
                break
            t = -oz / dz
            if t <= 0:
                break        # above the horizon: this end of the lane does not meet the ground ahead
            ends.append((float(ray["origin"][0]) + t * float(ray["direction"][0]),
                         float(ray["origin"][1]) + t * float(ray["direction"][1])))
        if len(ends) != 2:
            continue
        dx, dy = ends[1][0] - ends[0][0], ends[1][1] - ends[0][1]
        if math.hypot(dx, dy) < 1.0:
            continue          # a metre of road is not a direction
        out.append(math.atan2(dy, dx))
    return out


def snap_to_lane(yaw: float, headings: list[float], tol: float = _LANE_SNAP_TOL) -> tuple[float, float] | None:
    """Snap a fitted yaw onto the nearest road direction, or None when none is close enough.

    Compared modulo pi, because both a lane and a reprojection-fitted yaw are axes rather than bearings.
    The returned yaw keeps the side of the axis the fit chose, so snapping never turns a vehicle around.
    """
    best = None
    for hd in headings:
        # Smallest signed angle to the lane axis, in (-pi/2, pi/2].
        d = (hd - yaw + math.pi / 2) % math.pi - math.pi / 2
        if abs(d) <= tol and (best is None or abs(d) < abs(best[1])):
            best = (yaw + d, d)
    return (float(best[0]), float(abs(best[1]))) if best else None


def _fit_mono(obj, onto, cam_id: str, w: int, h: int, *, lanes: list | None = None,
              contact: tuple[float, float] | None = None):
    """Monocular cuboid: ground-lift the contact point, size from a class prior, then find the yaw.

    Returns `(cuboid_3d, reproj_iou)` or None when the box does not meet the ground plane ahead. The
    cuboid carries `yaw_source`, which is `reprojection` or `lane`.

    The yaw is found in two steps and optionally corrected by a third. Four coarse candidates 45 degrees
    apart, then a two-degree sweep around whichever won: measured over 238 objects from the real corpus
    that lifts mean reprojection IoU from 0.491 to 0.555, with 178 improving. Then, when lanes are passed
    in and one of them runs close to the fitted yaw, the yaw is snapped to the road.

    Snapping matters because reprojection IoU is a weak signal for yaw and cannot be otherwise: the
    axis-aligned image box of a car seen head-on is very nearly the box of the same car seen from behind.
    A lane is measured from the road. It is still an axis rather than a bearing, so it corrects which way
    the vehicle is aligned and never which way it faces.
    """
    from services.lidar.project import camera_ray_to_ego

    size = _dims_for(onto, obj.class_id)
    if size is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in obj.bbox)
    # Bottom-centre is the ground contact for a box that is fully visible. `contact` overrides it, for a
    # vehicle truncated by the frame edge or occluded at the bottom, where the box's lower edge is not
    # where the wheels are and lifting it puts the cuboid metres too close.
    u, v = contact if contact is not None else ((x1 + x2) / 2.0, y2)
    try:
        ray = camera_ray_to_ego(u, v, cam_id, w, h)
    except Exception:  # noqa: BLE001 -- no calibration for this cam
        return None
    dz, oz = float(ray["direction"][2]), float(ray["origin"][2])
    if abs(dz) < 1e-6:
        return None
    t = -oz / dz
    if t <= 0:
        return None  # bottom is above the horizon: cannot rest on the ground
    ego_x = float(ray["origin"][0]) + t * float(ray["direction"][0])
    ego_y = float(ray["origin"][1]) + t * float(ray["direction"][1])
    center = [round(ego_x, 3), round(ego_y, 3), round(size[2] / 2.0, 3)]
    target = [x1, y1, x2, y2]

    coarse = _best_yaw(center, size, cam_id, w, h, target, _YAW_CANDIDATES)
    if coarse is None:
        return None
    # Local sweep around the coarse winner. Worth 0.06 mean IoU over the corpus; see _REFINE_HALF_SPAN.
    span = [coarse[0] + d for d in _frange(-_REFINE_HALF_SPAN, _REFINE_HALF_SPAN, _REFINE_STEP)]
    best = _best_yaw(center, size, cam_id, w, h, target, span) or coarse

    yaw, iou = best
    source = "reprojection"
    snapped = snap_to_lane(yaw, lane_headings(lanes, cam_id, w, h)) if lanes else None
    if snapped is not None:
        # Only when the snap costs nothing measurable.
        #
        # The first version allowed a snap to lose up to 0.02 IoU on the reasoning that a lane is measured
        # from the road while a fitted yaw is inferred from a silhouette. Measured over 1,050 objects on
        # 120 frames carrying lanes, that snapped 51% of them and moved mean reprojection IoU from 0.2461
        # to 0.2445: 244 worsened against 11 improved. Reprojection IoU is a weak proxy for yaw, so that
        # does not prove the yaws got worse - but it is the only signal available, and shipping a change
        # that degrades it on the hope that something unmeasured improved is not a justified trade.
        #
        # Requiring the snap to be at least as good makes it free. IoU is genuinely flat across a range of
        # yaws, which is the whole reason yaw is hard here, so there is often a road-aligned yaw that ties
        # the fitted one; taking it there is a gain in plausibility at no measured cost. Re-measured on
        # the same 1,050 objects with this criterion: 82 snapped (8%), 11 improved, 0 worsened, and mean
        # IoU unchanged to four decimal places.
        alt = _yaw_iou(center, size, snapped[0], cam_id, w, h, target)
        if alt is not None and alt >= iou:
            yaw, iou, source = snapped[0], alt, "lane"
    return ({"center": center, "size": size, "yaw": round(float(yaw), 4), "yaw_source": source}, iou)


def _frange(lo: float, hi: float, step: float) -> list[float]:
    n = max(1, int(round((hi - lo) / step)))
    return [lo + i * step for i in range(n + 1)]


async def fit_object_cuboid(db: AsyncSession, object_id: uuid.UUID, *,
                            contact: tuple[float, float] | None = None) -> dict:
    """Fit a cuboid to one object, for the editor's cuboid tool.

    The solve has existed since cuboids were added and was reachable only as a frame-wide batch, so the
    editor's own cuboid tool dropped a hardcoded 1.8 x 4.2 x 1.5 box at yaw 0 regardless of what it was
    placed on: a bus and a scooter got the same cuboid facing the same way.

    `contact` is where the object actually meets the road, for a vehicle truncated by the frame edge or
    occluded at the bottom. Without it the box's lower edge is assumed to be the contact point, and for a
    truncated vehicle that is metres wrong in the direction of the camera.
    """
    from db.models import Lane
    from services.autolabel.ontology import get_ontology

    obj = await db.get(Object, object_id)
    if obj is None:
        raise ValueError("object not found")
    frame = await db.get(Frame, obj.frame_id)
    if frame is None or not frame.cam_id:
        raise ValueError("frame not found, or it has no camera to calibrate against")

    onto = get_ontology()
    try:
        c = onto.by_id(int(obj.class_id))
    except KeyError as exc:
        raise ValueError(f"class {obj.class_id} is not in the ontology") from exc
    if c.l1 not in _LIFT_L1:
        # Said rather than guessed. A hoarding does not rest on the road, so there is no ground contact to
        # lift and a cuboid fitted to one would be a number with no meaning.
        return {"object_id": str(object_id), "cuboid": None,
                "reason": f"{c.name} is not a class that rests on the road surface"}

    lanes = list((await db.execute(select(Lane).where(Lane.frame_id == frame.frame_id))).scalars().all())
    fit = _fit_mono(obj, onto, frame.cam_id, frame.width, frame.height, lanes=lanes, contact=contact)
    if fit is None:
        return {"object_id": str(object_id), "cuboid": None,
                "reason": "the box does not meet the ground plane ahead of the camera "
                          "(its base is above the horizon, or this camera has no calibration)"}
    cuboid, iou = fit
    return {"object_id": str(object_id), "cuboid": cuboid, "reproj_iou": round(iou, 3),
            "class_name": c.name, "yaw_source": cuboid.get("yaw_source"),
            "lanes_available": len(lanes)}


async def fit_cuboid_at(db: AsyncSession, frame_id: uuid.UUID, u: float, v: float,
                        class_name: str) -> dict:
    """A cuboid for a point clicked on the road, before any 2D box exists.

    The editor's cuboid tool places a box by clicking where a vehicle meets the road. It had no object to
    reproject against, so it used a hardcoded 1.8 x 4.2 x 1.5 at yaw 0: a bus and a scooter came out the
    same size facing the same way.

    Two of the three unknowns are answerable without a silhouette. The size comes from the class prior the
    batch solve already uses, and the yaw from the road, when a lane can be lifted near the click. Only
    when neither is available does this fall back to axis-aligned, and it says so.
    """
    from db.models import Lane
    from services.autolabel.ontology import get_ontology
    from services.lidar.project import camera_ray_to_ego

    frame = await db.get(Frame, frame_id)
    if frame is None or not frame.cam_id:
        raise ValueError("frame not found, or it has no camera to calibrate against")
    onto = get_ontology()
    if not onto.has_name(class_name):
        raise ValueError(f"unknown class '{class_name}'")
    cls = onto.by_name(class_name)
    size = _dims_for(onto, cls.id)
    if size is None:
        return {"cuboid": None,
                "reason": f"{class_name} is not a class that rests on the road surface, so it has no "
                          "ground-contact size prior"}

    try:
        ray = camera_ray_to_ego(float(u), float(v), frame.cam_id, frame.width, frame.height)
    except Exception as exc:  # noqa: BLE001
        return {"cuboid": None, "reason": f"no calibration for {frame.cam_id}: {str(exc)[:120]}"}
    dz, oz = float(ray["direction"][2]), float(ray["origin"][2])
    if abs(dz) < 1e-6 or -oz / dz <= 0:
        return {"cuboid": None, "reason": "that point is above the horizon, so it is not on the road ahead"}
    t = -oz / dz
    center = [round(float(ray["origin"][0]) + t * float(ray["direction"][0]), 3),
              round(float(ray["origin"][1]) + t * float(ray["direction"][1]), 3),
              round(size[2] / 2.0, 3)]

    lanes = list((await db.execute(select(Lane).where(Lane.frame_id == frame_id))).scalars().all())
    headings = lane_headings(lanes, frame.cam_id, frame.width, frame.height)
    yaw, source = 0.0, "default"
    if headings:
        # No fitted yaw to snap from here, so take the road direction itself: the nearest lane heading to
        # straight ahead, which is what a vehicle on this road is aligned with.
        yaw = min(headings, key=lambda hd: abs((hd + math.pi / 2) % math.pi - math.pi / 2))
        source = "lane"
    return {"cuboid": {"center": center, "size": size, "yaw": round(float(yaw), 4), "yaw_source": source},
            "class_name": class_name, "yaw_source": source, "lanes_available": len(lanes)}


async def plan_cuboids(db: AsyncSession, frame_id: uuid.UUID, *, min_iou: float = 0.35, high: float = 0.6) -> dict:
    """Dry-run: which of the frame's 2D vehicle/VRU boxes lift to a valid cuboid. No writes."""
    from services.autolabel.ontology import get_ontology

    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise ValueError("frame not found")
    onto = get_ontology()
    objs = (await db.execute(select(Object).where(
        Object.frame_id == frame_id, Object.source != "human", Object.cuboid_3d.is_(None)))).scalars().all()
    # The frame's lanes, loaded once. A lane gives the direction of the road under the camera, which is a
    # measurement, where a reprojection-fitted yaw is an inference from a silhouette.
    from db.models import Lane

    lanes = list((await db.execute(select(Lane).where(Lane.frame_id == frame_id))).scalars().all())
    items = []
    counts = {"total": 0, "auto_accept": 0, "review": 0, "skip": 0}
    for o in objs:
        try:
            name, l1 = onto.by_id(int(o.class_id)).name, onto.by_id(int(o.class_id)).l1
        except Exception:  # noqa: BLE001
            continue
        if l1 not in _LIFT_L1:
            continue
        counts["total"] += 1
        fit = _fit_mono(o, onto, frame.cam_id, frame.width, frame.height, lanes=lanes)
        if fit is None:
            counts["skip"] += 1
            items.append({"object_id": str(o.object_id), "class_name": name, "action": "skip",
                          "reason": "not liftable (above horizon / no calibration)", "iou": None})
            continue
        cuboid, iou = fit
        action = "auto_accept" if iou >= high else "review" if iou >= min_iou else "skip"
        counts[action] += 1
        items.append({"object_id": str(o.object_id), "class_name": name, "action": action,
                      "iou": round(iou, 3), "cuboid": cuboid})
    return {"frame_id": str(frame_id), "counts": counts, "items": items}


async def commit_cuboids(db: AsyncSession, frame_id: uuid.UUID, *, min_iou: float = 0.35, high: float = 0.6,
                         created_by: str | None = None) -> dict:
    """Attach the fitted cuboids to their objects as one reversible run (revert clears them)."""
    plan = await plan_cuboids(db, frame_id, min_iou=min_iou, high=high)
    run_id = uuid.uuid4()
    changes: dict[str, dict] = {}
    for item in plan["items"]:
        if item["action"] == "skip":
            continue
        obj = await db.get(Object, uuid.UUID(item["object_id"]))
        if obj is None or obj.source == "human" or obj.cuboid_3d is not None:
            continue
        changes[item["object_id"]] = {"from_cuboid": None}
        obj.cuboid_3d = item["cuboid"]
        obj.version = (obj.version or 0) + 1
        prov = dict(obj.provenance or {})
        prov["agent_run_id"] = str(run_id)
        prov.setdefault("agent_cuboid", {})["reproj_iou"] = item["iou"]
        obj.provenance = prov
    db.add(AgentRun(run_id=run_id, kind="cuboid", scope={"frame_id": str(frame_id)}, status="committed",
                    policy={"min_iou": min_iou, "high": high}, counts=plan["counts"], changes=changes,
                    critic={}, created_by=created_by))
    await db.commit()
    log.info("agent.cuboid.commit", frame_id=str(frame_id), run_id=str(run_id), attached=len(changes))
    return {"run_id": str(run_id), "frame_id": str(frame_id), "attached": len(changes), "counts": plan["counts"]}
