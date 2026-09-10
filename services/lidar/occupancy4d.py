"""Occupancy with scene flow: what space is filled around the vehicle, and how that space is moving.

A cuboid list answers "which objects did the detector find". It has no answer for the space between them,
and on an Indian road the space between them is where the sand pile, the unlabelled handcart and the
object the detector missed all are. An occupancy grid answers per cubic metre, including the cubic metres
nothing was labelled in.

**The fourth dimension is the point.** A static grid cannot separate a parked car from one reversing
toward the ego at the instant the frame was taken, and that difference is the whole of planning. Each
occupied voxel carries a velocity taken from the 3D track whose cuboid contains it.

**A voxel no track claims holds zero, and the row says how many those were.** Zero is the right value:
most of a road scene genuinely is static, and inventing motion for unclaimed space would be worse than
assuming none. But an assumed zero and a measured zero are different facts, so `flow_voxels` counts the
occupied voxels a track actually spoke for. A grid where that number is small is a grid mostly made of
assumption, and nothing should plan against it without knowing that.

**Placement needs a pose and says when it lacks one.** Two grids from two instants are only comparable if
both are in the same frame, which is what `ego_pose` (0112) is for. Without a pose the grid is built in
the ego frame of its own instant, `ego_pose_ts` is null, and a consumer stacking it against the next one
can tell.
"""

from __future__ import annotations

import io
import math
import uuid
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import EgoPose, Frame, OccupancyGrid, PointCloud

log = get_logger("occupancy4d")

KIND = "occupancy_build"
# Half a metre. Small enough that a motorcycle is several voxels and a gap between vehicles is visible,
# large enough that a 100 by 100 metre window is not a gigabyte.
VOXEL_M = 0.5
# The window around the vehicle, in metres: forward, lateral each way, and vertical. Behind matters less
# than ahead for planning, and the pseudo-LiDAR lift only sees what the cameras saw anyway.
BOUNDS = (-20.0, -30.0, -2.0, 60.0, 30.0, 4.0)
# Points in a voxel before it counts as occupied. One point is noise on a monocular depth cloud, where a
# single bad pixel back-projects to a floating speck metres from anything.
MIN_POINTS = 3
# Grids per commit. The same batch-by-batch rule as everything else here: a window of a thousand frames
# must not be one transaction.
GRID_BATCH = 25


@dataclass(frozen=True)
class FlowSource:
    """One track's contribution to the flow field at an instant: where it is and how fast it moves."""

    center: tuple[float, float, float]
    dims: tuple[float, float, float]
    yaw: float
    velocity: tuple[float, float, float]


def voxel_indices(points: np.ndarray, origin: np.ndarray, dims: tuple[int, int, int],
                  voxel_m: float, min_points: int = MIN_POINTS) -> np.ndarray:
    """The occupied voxel indices of a cloud, as an (N, 3) integer array.

    Vectorised and deduplicated: a cloud is up to millions of points and the interesting quantity is the
    set of cells, not the points.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    vi = np.floor((pts - origin) / voxel_m).astype(np.int64)
    keep = np.ones(len(vi), dtype=bool)
    for axis in range(3):
        keep &= (vi[:, axis] >= 0) & (vi[:, axis] < dims[axis])
    vi = vi[keep]
    if vi.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    uniq, counts = np.unique(vi, axis=0, return_counts=True)
    return uniq[counts >= min_points].astype(np.int32)


def _in_box(points_world: np.ndarray, src: FlowSource) -> np.ndarray:
    """Which of these world points fall inside an oriented cuboid. Yaw only, as everything 3D here is."""
    if points_world.size == 0:
        return np.zeros(0, dtype=bool)
    c = np.asarray(src.center, dtype=np.float64)
    rel = points_world - c
    ca, sa = math.cos(-src.yaw), math.sin(-src.yaw)
    # Rotate into the box's own axes, then a simple half-extent test.
    lx = rel[:, 0] * ca - rel[:, 1] * sa
    ly = rel[:, 0] * sa + rel[:, 1] * ca
    lz = rel[:, 2]
    hx, hy, hz = (d / 2.0 for d in src.dims)
    return (np.abs(lx) <= hx) & (np.abs(ly) <= hy) & (np.abs(lz) <= hz)


def assign_flow(voxels: np.ndarray, origin: np.ndarray, voxel_m: float,
                sources: list[FlowSource]) -> tuple[np.ndarray, int]:
    """A velocity per occupied voxel, and how many of them a track actually spoke for.

    Voxel centres rather than corners, because a cuboid edge cutting a voxel in half should claim it when
    most of the voxel is inside. Later sources win a contested voxel, which is the arbitrary half of an
    arbitrary choice; two cuboids overlapping is itself a labelling problem and the flow field is not
    where to fix it.
    """
    flow = np.zeros((len(voxels), 3), dtype=np.float32)
    if len(voxels) == 0 or not sources:
        return flow, 0
    centres = origin + (voxels.astype(np.float64) + 0.5) * voxel_m
    claimed = np.zeros(len(voxels), dtype=bool)
    for src in sources:
        inside = _in_box(centres, src)
        if not inside.any():
            continue
        flow[inside] = np.asarray(src.velocity, dtype=np.float32)
        claimed |= inside
    return flow, int(claimed.sum())


def pack_grid(voxels: np.ndarray, flow: np.ndarray, dims: tuple[int, int, int]) -> bytes:
    """The grid as a compressed npz of sparse indices and their flow.

    Sparse rather than dense: a 160 by 120 by 12 grid is 230,400 voxels and a road scene occupies a few
    thousand of them, so a dense uint8 array would be almost entirely zeros and the flow field would be
    three float16 arrays of the same shape. Indices plus values is the same information an order of
    magnitude smaller.
    """
    buf = io.BytesIO()
    np.savez_compressed(buf, voxels=voxels.astype(np.int32),
                        flow=flow.astype(np.float16), dims=np.asarray(dims, dtype=np.int32))
    return buf.getvalue()


def unpack_grid(data: bytes) -> dict:
    """The inverse of `pack_grid`, for a consumer that wants the voxels back."""
    with np.load(io.BytesIO(data)) as z:
        return {"voxels": z["voxels"], "flow": z["flow"].astype(np.float32),
                "dims": tuple(int(d) for d in z["dims"])}


async def _pose_at(db: AsyncSession, session_id: uuid.UUID, ts_ns: int) -> EgoPose | None:
    return (await db.execute(select(EgoPose).where(
        EgoPose.session_id == session_id, EgoPose.ts_ns == ts_ns))).scalar_one_or_none()


def _ego_to_world(points: np.ndarray, pose: EgoPose | None) -> np.ndarray:
    """Ego-frame points in the session frame, or unchanged when there is no pose to place them with."""
    if pose is None or points.size == 0:
        return points
    yaw = 2.0 * math.atan2(pose.qz, pose.qw)
    ca, sa = math.cos(yaw), math.sin(yaw)
    out = np.empty_like(points)
    out[:, 0] = pose.x + points[:, 0] * ca - points[:, 1] * sa
    out[:, 1] = pose.y + points[:, 0] * sa + points[:, 1] * ca
    out[:, 2] = pose.z + points[:, 2]
    return out


async def _flow_sources(db: AsyncSession, session_id: uuid.UUID, ts_ns: int,
                        window_ns: int) -> list[FlowSource]:
    """The 3D tracks that have a cuboid near this instant, with the velocity their trajectory implies.

    Velocity from the two nearest trajectory points rather than from a stored field, because `track_3d`
    keeps a trajectory and not a speed, and differencing the trajectory is what the trajectory is for.
    """
    from db.models import Track3D

    tracks = (await db.execute(select(Track3D).where(
        Track3D.session_id == session_id,
        Track3D.first_ts_ns <= ts_ns + window_ns,
        Track3D.last_ts_ns >= ts_ns - window_ns))).scalars().all()
    out: list[FlowSource] = []
    for t in tracks:
        pts = ((t.trajectory or {}).get("points") or [])
        if len(pts) < 2:
            continue
        # The two samples bracketing this instant.
        pts = sorted(pts, key=lambda p: int(p.get("ts_ns", 0)))
        idx = min(range(len(pts)), key=lambda i: abs(int(pts[i].get("ts_ns", 0)) - ts_ns))
        j = idx + 1 if idx + 1 < len(pts) else idx - 1
        if j < 0:
            continue
        a, b = pts[min(idx, j)], pts[max(idx, j)]
        dt = (int(b.get("ts_ns", 0)) - int(a.get("ts_ns", 0))) / 1e9
        if dt <= 0:
            continue
        pa, pb = np.asarray(a["xyz"], dtype=float), np.asarray(b["xyz"], dtype=float)
        vel = tuple(float(v) for v in (pb - pa) / dt)
        centre = tuple(float(v) for v in pts[idx]["xyz"])
        dims = tuple(float(v) for v in (pts[idx].get("dims") or (4.0, 1.8, 1.5)))
        out.append(FlowSource(center=centre, dims=dims, yaw=float(pts[idx].get("yaw", 0.0)),
                              velocity=vel))
    return out


async def build_grid_for_cloud(db: AsyncSession, cloud: PointCloud, *, voxel_m: float = VOXEL_M,
                               bounds: tuple = BOUNDS, run_id: uuid.UUID | None = None) -> dict:
    """One cloud into one occupancy grid with its flow field. Returns what it wrote or why it did not."""
    from core.storage import get_object_store
    from services.lidar.ingest.store import load_cloud

    try:
        c = load_cloud(cloud.cloud_uri)
    except Exception as exc:  # noqa: BLE001 - an unreadable cloud is a skipped grid, not a failed window
        return {"built": False, "reason": f"cloud unreadable: {str(exc)[:120]}"}
    xyz = np.asarray(getattr(c, "xyz", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
    if xyz.size == 0:
        return {"built": False, "reason": "the cloud has no points"}

    pose = await _pose_at(db, cloud.session_id, int(cloud.ts_ns))
    world = _ego_to_world(xyz, pose)

    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    # The window travels with the vehicle: its origin is the pose plus the ego-frame offset, so the grid
    # covers the road ahead rather than a fixed patch of the world the vehicle has already left.
    base = np.array([pose.x, pose.y, pose.z], dtype=np.float64) if pose is not None else np.zeros(3)
    origin = base + np.array([xmin, ymin, zmin], dtype=np.float64)
    dims = (max(1, int((xmax - xmin) / voxel_m)), max(1, int((ymax - ymin) / voxel_m)),
            max(1, int((zmax - zmin) / voxel_m)))

    voxels = voxel_indices(world, origin, dims, voxel_m)
    sources = await _flow_sources(db, cloud.session_id, int(cloud.ts_ns), window_ns=200_000_000)
    flow, claimed = assign_flow(voxels, origin, voxel_m, sources)

    frame_id = (await db.execute(select(Frame.frame_id).where(
        Frame.session_id == cloud.session_id, Frame.ts_ns == cloud.ts_ns).limit(1))).scalar_one_or_none()

    store = get_object_store()
    uri = store.put_bytes(f"occupancy/{cloud.session_id}/{cloud.ts_ns}.npz",
                          pack_grid(voxels, flow, dims), "application/octet-stream")
    source = "pseudo" if cloud.source == "pseudo" else "lidar"
    fields = {"frame_id": frame_id, "origin": [float(v) for v in origin], "voxel_m": float(voxel_m),
              "dims": list(dims), "grid_uri": uri,
              "ego_pose_ts": int(pose.ts_ns) if pose is not None else None,
              "occupied": int(len(voxels)), "flow_voxels": int(claimed), "run_id": run_id}

    # Upsert on the natural key, not on the primary key. `merge` keys on the primary key, and a new row
    # carries a fresh uuid, so rebuilding a session's grids raised a unique violation on
    # (session_id, ts_ns, source) rather than replacing them. A builder that cannot be re-run is one that
    # cannot be corrected, and this one had to be the moment real 3D tracks existed to give it a flow
    # field: the first pass built 128 grids whose flow was entirely an assumed zero.
    existing = (await db.execute(select(OccupancyGrid).where(
        OccupancyGrid.session_id == cloud.session_id, OccupancyGrid.ts_ns == int(cloud.ts_ns),
        OccupancyGrid.source == source))).scalars().first()
    if existing is not None:
        for k, v in fields.items():
            setattr(existing, k, v)
    else:
        db.add(OccupancyGrid(session_id=cloud.session_id, ts_ns=int(cloud.ts_ns), source=source,
                             **fields))
    return {"built": True, "ts_ns": int(cloud.ts_ns), "occupied": int(len(voxels)),
            "replaced": existing is not None,
            "flow_voxels": int(claimed), "tracks": len(sources),
            "placed": pose is not None, "dims": list(dims)}


async def build_occupancy_window(session_id: uuid.UUID, t0: int | None = None, t1: int | None = None, *,
                                 limit: int = 500, run_id: uuid.UUID | None = None,
                                 voxel_m: float = VOXEL_M) -> dict:
    """Every cloud in a session's time window into a grid, committed in batches.

    Returns the counts and, when the flow field is mostly assumption, says so rather than leaving the
    reader to divide two numbers and work it out.
    """
    from db.session import get_sessionmaker

    maker = get_sessionmaker()
    async with maker() as db:
        q = select(PointCloud).where(PointCloud.session_id == session_id).order_by(PointCloud.ts_ns)
        if t0 is not None:
            q = q.where(PointCloud.ts_ns >= t0)
        if t1 is not None:
            q = q.where(PointCloud.ts_ns <= t1)
        clouds = (await db.execute(q.limit(limit))).scalars().all()

    if not clouds:
        return {"session_id": str(session_id), "grids": 0,
                "reason": "no point cloud exists in this window; run the pseudo-LiDAR lift first"}

    built = skipped = occupied = flow_voxels = placed = 0
    reasons: dict[str, int] = {}
    async with maker() as db:
        for i, cloud in enumerate(clouds):
            res = await build_grid_for_cloud(db, cloud, voxel_m=voxel_m, run_id=run_id)
            if res["built"]:
                built += 1
                occupied += res["occupied"]
                flow_voxels += res["flow_voxels"]
                placed += 1 if res["placed"] else 0
            else:
                skipped += 1
                reasons[res["reason"]] = reasons.get(res["reason"], 0) + 1
            if (i + 1) % GRID_BATCH == 0:
                await db.commit()
        await db.commit()

    share = (flow_voxels / occupied) if occupied else None
    out = {"session_id": str(session_id), "grids": built, "skipped": skipped,
           "occupied_voxels": occupied, "flow_voxels": flow_voxels,
           "flow_share": round(share, 4) if share is not None else None,
           "grids_placed_by_pose": placed, "reasons": reasons}
    if share is not None and share < 0.05:
        # Said out loud rather than left as two numbers to divide. A grid whose flow is 3% real and 97%
        # assumed zero is a static grid wearing a velocity field.
        out["caveat"] = (f"only {share:.1%} of occupied space carries a velocity from a track; the rest "
                         f"is an assumed zero, so this is close to a static grid")
    if placed == 0 and built:
        out["caveat_pose"] = ("no grid was placed by an ego pose, so each sits in the ego frame of its "
                              "own instant and two of them cannot be stacked or compared")
    log.info("occupancy4d.window", **{k: v for k, v in out.items() if not isinstance(v, dict)})
    return out
