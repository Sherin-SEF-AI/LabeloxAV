"""A session window as a reconstruction sees it: camera frames with a world pose each, and the LiDAR around them.

Everything is in the session scene frame of services/lidar/aggregate/scene_frame.py: T_world_ego is the ego pose
at that instant times the lever arm from the pose unit to the ego (sensor) frame. A camera's pose follows from
its calibration, which maps an ego point to the optical frame as cam = (ego - t) @ R, so ego = R @ cam + t and
T_ego_cam = [[R, t], [0, 1]]. Placing the cameras and the LiDAR in the same frame by the same chain is what lets
a cuboid drawn in the reconstruction go to every scan and every frame through the label-once propagation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.origin import REAL

# A pose further than this from a frame's timestamp is a different instant (same rule as the scene frame).
POSE_MATCH_NS = 60_000_000


@dataclass
class WindowFrame:
    frame_id: uuid.UUID
    ts_ns: int
    cam_id: str
    img_uri: str
    width: int
    height: int
    K: np.ndarray                # 3x3 intrinsics at (width, height)
    T_world_cam: np.ndarray      # 4x4, optical frame (x right, y down, z forward) -> world


@dataclass
class Window:
    session_id: uuid.UUID
    cam_id: str
    frames: list[WindowFrame]
    scans: list = field(default_factory=list)   # ScenePose rows of the scans inside the window's time span
    cal_source: str = "nominal"
    summary: dict = field(default_factory=dict)

    @property
    def span_ns(self) -> tuple[int, int]:
        return (self.frames[0].ts_ns, self.frames[-1].ts_ns) if self.frames else (0, 0)


def camera_to_ego(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """T_ego_cam from a calibration's (R, t), where cam = (ego - t) @ R."""
    m = np.eye(4)
    m[:3, :3] = np.asarray(R, dtype=np.float64)
    m[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return m


def nearest_pose_index(pose_ts: np.ndarray, ts: int, tol_ns: int = POSE_MATCH_NS) -> int | None:
    """Index of the pose closest to `ts` within the tolerance, or None."""
    if len(pose_ts) == 0:
        return None
    i = int(np.searchsorted(pose_ts, ts))
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(pose_ts) and abs(int(pose_ts[j]) - int(ts)) <= tol_ns:
            if best is None or abs(int(pose_ts[j]) - int(ts)) < abs(int(pose_ts[best]) - int(ts)):
                best = j
    return best


async def session_cameras(db: AsyncSession, session_id: uuid.UUID) -> list[dict]:
    """The session's cameras with their real-frame count, largest first."""
    from sqlalchemy import func

    from db.models import Frame

    rows = (await db.execute(select(Frame.cam_id, func.count(), func.min(Frame.ts_ns), func.max(Frame.ts_ns))
                             .where(Frame.session_id == session_id, Frame.origin == REAL)
                             .group_by(Frame.cam_id).order_by(func.count().desc()))).all()
    return [{"cam_id": c, "frames": int(n), "first_ts_ns": int(a), "last_ts_ns": int(b)} for c, n, a, b in rows]


async def load_window(db: AsyncSession, session_id: uuid.UUID, *, cam_id: str | None = None,
                      start_ts_ns: int | None = None, n_frames: int = 40, stride: int = 1) -> Window:
    """The frames of one camera from `start_ts_ns` (default: the first), every `stride`-th, up to `n_frames`,
    each with its world pose; plus the scans inside that span. Frames without a pose are skipped and counted."""
    from db.models import EgoPose, Frame
    from db.models import Session as DbSession
    from services.calibration.resolve import resolve_calibration
    from services.lidar.aggregate.scene_frame import lever_arm, pose_matrix, session_scene_poses

    sess = await db.get(DbSession, session_id)
    if sess is None:
        raise LookupError("session not found")
    if cam_id is None:
        cams = await session_cameras(db, session_id)
        if not cams:
            raise LookupError("this session has no real camera frames")
        cam_id = cams[0]["cam_id"]
    q = (select(Frame.frame_id, Frame.ts_ns, Frame.img_uri, Frame.width, Frame.height)
         .where(Frame.session_id == session_id, Frame.cam_id == cam_id, Frame.origin == REAL)
         .order_by(Frame.ts_ns))
    if start_ts_ns is not None:
        q = q.where(Frame.ts_ns >= int(start_ts_ns))
    rows = (await db.execute(q.limit(max(1, n_frames) * max(1, stride)))).all()[::max(1, stride)]
    poses = (await db.execute(select(EgoPose).where(EgoPose.session_id == session_id)
                              .order_by(EgoPose.ts_ns))).scalars().all()
    pose_ts = np.array([p.ts_ns for p in poses], dtype=np.int64)
    arm = lever_arm(sess.sensors)
    frames: list[WindowFrame] = []
    no_pose = 0
    cal = None
    for fid, ts, uri, w, h in rows:
        j = nearest_pose_index(pose_ts, int(ts))
        if j is None:
            no_pose += 1
            continue
        if cal is None:
            # Committed here: resolve_calibration opens its own session, and this one must not sit idle in a
            # transaction while it does.
            await db.commit()
            cal = await resolve_calibration(session_id, cam_id, int(w), int(h))
        t_world_ego = pose_matrix(poses[j]) @ arm
        frames.append(WindowFrame(frame_id=fid, ts_ns=int(ts), cam_id=cam_id, img_uri=uri, width=int(w),
                                  height=int(h), K=cal.K(),
                                  T_world_cam=t_world_ego @ camera_to_ego(cal.R(), cal.t())))
    scans = []
    if frames:
        all_scans, _ = await session_scene_poses(db, session_id)
        lo, hi = frames[0].ts_ns - POSE_MATCH_NS, frames[-1].ts_ns + POSE_MATCH_NS
        scans = [s for s in all_scans if lo <= s.ts_ns <= hi]
    return Window(session_id=session_id, cam_id=cam_id, frames=frames, scans=scans,
                  cal_source=cal.source if cal else "none",
                  summary={"frames_requested": len(rows), "frames_posed": len(frames), "frames_without_pose": no_pose,
                           "scans": len(scans), "ego_poses": len(poses), "calibration": cal.source if cal else None})


def frustum_mask(xyz: np.ndarray, frames: list[WindowFrame], max_depth: float = 80.0) -> np.ndarray:
    """Points inside at least one frame's view (in front, within the image, nearer than `max_depth`)."""
    keep = np.zeros(len(xyz), dtype=bool)
    for f in frames:
        T = np.linalg.inv(f.T_world_cam)
        cam = xyz @ T[:3, :3].T + T[:3, 3]
        z = cam[:, 2]
        ok = (z > 0.5) & (z < max_depth)
        zs = np.where(ok, z, 1.0)
        u = f.K[0, 0] * cam[:, 0] / zs + f.K[0, 2]
        v = f.K[1, 1] * cam[:, 1] / zs + f.K[1, 2]
        keep |= ok & (u >= 0) & (u < f.width) & (v >= 0) & (v < f.height)
    return keep


def colourise(xyz: np.ndarray, frames: list[WindowFrame], images: list[np.ndarray], *,
              zbuf_scale: int = 4, tol_m: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """A colour per point from the frames that see it unoccluded, weighted towards the nearest view.

    Visibility is a z-buffer per frame at 1/`zbuf_scale` resolution: a point is coloured by a frame only when it
    is within `tol_m` (plus 2% of its depth) of the nearest point on its pixel, so the road behind a car does not
    take the car's colour. Returns (rgb float32 0..1, seen bool).
    """
    acc = np.zeros((len(xyz), 3), dtype=np.float64)
    wsum = np.zeros(len(xyz), dtype=np.float64)
    for f, img in zip(frames, images, strict=True):
        T = np.linalg.inv(f.T_world_cam)
        cam = xyz @ T[:3, :3].T + T[:3, 3]
        z = cam[:, 2]
        front = z > 0.5
        zs = np.where(front, z, 1.0)
        u = f.K[0, 0] * cam[:, 0] / zs + f.K[0, 2]
        v = f.K[1, 1] * cam[:, 1] / zs + f.K[1, 2]
        h, w = img.shape[:2]
        ok = front & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            continue
        gw, gh = max(1, w // zbuf_scale), max(1, h // zbuf_scale)
        cell = (v[idx] // zbuf_scale).astype(np.int64).clip(0, gh - 1) * gw + \
            (u[idx] // zbuf_scale).astype(np.int64).clip(0, gw - 1)
        zbuf = np.full(gw * gh, np.inf)
        np.minimum.at(zbuf, cell, z[idx])
        vis = z[idx] <= zbuf[cell] + tol_m + 0.02 * z[idx]
        idx = idx[vis]
        px = img[v[idx].astype(np.int64), u[idx].astype(np.int64)].astype(np.float64) / 255.0
        wt = 1.0 / np.maximum(z[idx], 1.0)
        acc[idx] += px * wt[:, None]
        wsum[idx] += wt
    seen = wsum > 0
    rgb = np.full((len(xyz), 3), 0.5, dtype=np.float64)
    rgb[seen] = acc[seen] / wsum[seen, None]
    return rgb.astype(np.float32), seen


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """T_world_cam for an optical-frame camera at `eye` looking at `target` (x right, y down, z forward)."""
    z = target - eye
    z = z / (np.linalg.norm(z) or 1.0)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x = x / (np.linalg.norm(x) or 1.0)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, eye
    return m


def novel_poses(f: WindowFrame, world_up: np.ndarray) -> dict[str, np.ndarray]:
    """Views a driving camera never took, around one frame: 1.5 m to each side, and raised 3 m looking down.

    Each is a pose the trajectory does not contain, so what a reconstruction renders there is what it actually
    learned about the geometry, not a replay of a training image.
    """
    T = f.T_world_cam
    right, fwd = T[:3, 0], T[:3, 2]
    eye = T[:3, 3]
    out = {}
    for name, d in (("left_1.5m", -1.5), ("right_1.5m", 1.5)):
        m = T.copy()
        m[:3, 3] = eye + right * d
        out[name] = m
    up = world_up / (np.linalg.norm(world_up) or 1.0)
    fwd_flat = fwd - up * float(fwd @ up)
    fwd_flat = fwd_flat / (np.linalg.norm(fwd_flat) or 1.0)
    raised = eye + up * 3.0 - fwd_flat * 2.0
    out["raised_3m"] = look_at(raised, eye + fwd_flat * 14.0, up)
    return out


def bev_view(xyz: np.ndarray, heading_xy: np.ndarray | None = None, max_px: int = 1400,
             margin_m: float = 4.0) -> dict:
    """An orthographic top-down view of the points: T_world_cam and K where fx = fy = pixels per metre.

    The camera looks straight down; the image's up direction is `heading_xy` (the way the drive went), so the
    road runs up the screen whatever its compass bearing. A pixel (u, v) is the world point
    origin + right * u / ppm - heading * v / ppm, which is how the page turns a click into a world (x, y).
    """
    h = np.array([1.0, 0.0]) if heading_xy is None else np.asarray(heading_xy, dtype=np.float64)[:2]
    h = h / (np.linalg.norm(h) or 1.0)
    fwd = np.array([h[0], h[1], 0.0])
    right = np.array([h[1], -h[0], 0.0])
    a = xyz @ right
    b = xyz @ -fwd
    a0, a1 = float(a.min()) - margin_m, float(a.max()) + margin_m
    b0, b1 = float(b.min()) - margin_m, float(b.max()) + margin_m
    ppm = float(max_px / max(a1 - a0, b1 - b0, 1.0))
    w, hgt = int(np.ceil((a1 - a0) * ppm)), int(np.ceil((b1 - b0) * ppm))
    top = float(np.percentile(xyz[:, 2], 99.5)) + 20.0
    m = np.eye(4)
    m[:3, 0] = right
    m[:3, 1] = -fwd
    m[:3, 2] = [0.0, 0.0, -1.0]
    m[:3, 3] = right * a0 - fwd * b0 + np.array([0.0, 0.0, top])
    K = np.array([[ppm, 0.0, 0.0], [0.0, ppm, 0.0], [0.0, 0.0, 1.0]])
    return {"T_world_cam": m, "K": K, "width": w, "height": hgt, "ppm": ppm, "top_z": top,
            "heading": [float(h[0]), float(h[1])]}
