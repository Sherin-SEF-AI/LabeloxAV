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


def _fit_mono(obj, onto, cam_id: str, w: int, h: int):
    """Monocular cuboid: ground-lift the box's bottom-centre, size from class prior, pick the best-reprojecting
    yaw. Returns (cuboid_3d dict, reproj_iou) or None if the box does not touch the ground plane ahead."""
    from services.lidar.boxes import project_cuboid
    from services.lidar.project import camera_ray_to_ego

    size = _dims_for(onto, obj.class_id)
    if size is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in obj.bbox)
    u, v = (x1 + x2) / 2.0, y2  # bottom-centre = ground contact
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
    best = None
    for yaw in _YAW_CANDIDATES:
        dims = [size[1], size[0], size[2]]  # project_cuboid wants [length, width, height]
        proj = project_cuboid(center, dims, yaw, cam_id, w, h)
        rb = _reproj_box(proj)
        if rb is None:
            continue
        iou = _iou2d(rb, [x1, y1, x2, y2])
        if best is None or iou > best[1]:
            best = ({"center": center, "size": size, "yaw": round(yaw, 4)}, iou)
    return best


async def plan_cuboids(db: AsyncSession, frame_id: uuid.UUID, *, min_iou: float = 0.35, high: float = 0.6) -> dict:
    """Dry-run: which of the frame's 2D vehicle/VRU boxes lift to a valid cuboid. No writes."""
    from services.autolabel.ontology import get_ontology

    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise ValueError("frame not found")
    onto = get_ontology()
    objs = (await db.execute(select(Object).where(
        Object.frame_id == frame_id, Object.source != "human", Object.cuboid_3d.is_(None)))).scalars().all()
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
        fit = _fit_mono(o, onto, frame.cam_id, frame.width, frame.height)
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
