"""Recovering where the vehicle was, from whatever the session actually carries.

`calyx/ego_propagate.ego_transform` refuses on every session in this corpus, and correctly: with GNSS on
3 frames of 41,752 there is nothing to derive a rigid transform from. This module is what fills the gap,
and it fills it honestly rather than by pretending.

Two sources, and the table records which one answered.

**GNSS and speed** (`source='gnss_imu'`, `measured=True`). Where fixes exist, position comes from the
fixes in a session-local ENU frame and heading from `derive_ego_state`. This is an instrument observing
position, so it is a measurement, and on this corpus it will answer for almost nothing.

**Monocular visual odometry** (`source='visual'`, `measured=False`). ORB features between consecutive
frames, an essential matrix against the resolved intrinsics, and `recoverPose` for rotation and a unit
translation. Monocular VO cannot recover scale from images alone, which is the whole difficulty, and the
two ways out of it are both here:

  1. The metric depth already computed for the session's pseudo-LiDAR clouds. Where a cloud exists at a
     frame's timestamp, the median forward distance of the matched features gives the baseline in metres.
  2. The known camera height above the road. The ground plane is at a fixed height in the ego frame, so
     the apparent motion of ground-plane features fixes the baseline. This is the fallback and it is
     weaker, because it assumes the road is flat and the mount height is right.

Where neither is available the direction is still recovered and the step is written with `speed_mps`
null and a low `quality`, so a consumer can use the rotation and refuse the translation. It is never
scaled by a guess.

`measured=False` on every visual row is the load-bearing part. A trajectory recovered from pixels is not
a trajectory anybody measured, and a downstream consumer that needs a real one must be able to tell.
"""

from __future__ import annotations

import math
import uuid

import cv2
import numpy as np
from geoalchemy2 import Geometry
from sqlalchemy import cast, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import EgoPose, Frame, PointCloud
from db.models import Session as DbSession

log = get_logger("ego_pose")

GNSS = "gnss_imu"
VISUAL = "visual"
FUSED = "fused"

# ORB budget per frame. Enough to survive an Indian road scene where most of the image is moving traffic,
# small enough that a 1,000-frame session is minutes rather than hours.
ORB_FEATURES = 1500
# A pair with fewer surviving matches than this has not seen the same scene twice, and an essential
# matrix fitted to it is noise with a confident-looking shape.
MIN_MATCHES = 40
# RANSAC threshold in pixels for the essential matrix.
RANSAC_PX = 1.0
RANSAC_CONF = 0.999
# A step longer than this between consecutive frames is not a step, it is a gap in the recording, and
# integrating across it would produce a trajectory that never happened.
MAX_STEP_S = 2.0
# Plausible ground speed ceiling. A recovered baseline above it is a scale failure, not a fast car.
MAX_SPEED_MPS = 45.0
# Frames per commit. The whole point of this program is that nothing runs as one unbounded block.
POSE_BATCH = 200

_R_EARTH_M = 6_371_000.0


def enu_offset(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """Metres east and north of a local origin, on the small-angle approximation.

    Good to well under a metre over the few kilometres a driving session covers, which is far below the
    accuracy of anything else here. A full geodetic solution would be false precision on a trajectory
    whose rotation comes from ORB features.
    """
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    east = dlon * _R_EARTH_M * math.cos(math.radians(lat0))
    north = dlat * _R_EARTH_M
    return east, north


def yaw_to_quat(yaw_rad: float) -> tuple[float, float, float, float]:
    """A yaw-only rotation as (qw, qx, qy, qz).

    Yaw only, because roll and pitch are not observable from any source this corpus has: GNSS gives no
    attitude and monocular VO's pitch is entangled with the scale it cannot recover. Writing a fabricated
    roll would make the pose look more complete than it is.
    """
    half = yaw_rad / 2.0
    return math.cos(half), 0.0, 0.0, math.sin(half)


def quat_yaw(qw: float, qz: float) -> float:
    return 2.0 * math.atan2(qz, qw)


def _detector():
    return cv2.ORB_create(nfeatures=ORB_FEATURES)


def _match(desc_a, desc_b) -> list:
    if desc_a is None or desc_b is None or len(desc_a) < 8 or len(desc_b) < 8:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return sorted(bf.match(desc_a, desc_b), key=lambda m: m.distance)


def relative_pose(kp_a, desc_a, kp_b, desc_b, K: np.ndarray) -> dict | None:
    """Rotation and a unit-length translation direction between two views, or None when unrecoverable.

    Returns {"R", "t_unit", "yaw", "inliers", "pts_a", "pts_b"}. The translation is a direction only: a
    single camera cannot see scale, and returning a length here would be inventing one.
    """
    matches = _match(desc_a, desc_b)
    if len(matches) < MIN_MATCHES:
        return None
    pts_a = np.float64([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float64([kp_b[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(pts_a, pts_b, K, method=cv2.RANSAC, prob=RANSAC_CONF,
                                   threshold=RANSAC_PX)
    if E is None or E.shape != (3, 3):
        return None
    n_in, R, t, pose_mask = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)
    if n_in < MIN_MATCHES // 2:
        return None
    keep = (pose_mask.ravel() > 0)
    # Camera optical frame is x right, y down, z forward, so a yaw in the ego frame is a rotation about
    # the optical y axis, and its sign flips going from a down axis to an up one.
    yaw = -math.atan2(float(R[0, 2]), float(R[2, 2]))
    return {"R": R, "t_unit": t.reshape(3), "yaw": yaw, "inliers": int(n_in),
            "pts_a": pts_a[keep], "pts_b": pts_b[keep]}


def scale_from_depth(depth_m: np.ndarray, pts_a: np.ndarray, t_unit: np.ndarray) -> float | None:
    """Baseline in metres from a metric depth map and the matched features, or None.

    The forward component of the unit translation, multiplied by the median depth of the features that
    survived the pose fit, is the distance the camera moved along its optical axis in the units the depth
    map is in. Median rather than mean because a single feature on a passing truck is a large outlier and
    half the scene in this corpus is moving traffic.
    """
    if depth_m is None or len(pts_a) < MIN_MATCHES // 2:
        return None
    h, w = depth_m.shape[:2]
    xs = np.clip(pts_a[:, 0].astype(int), 0, w - 1)
    ys = np.clip(pts_a[:, 1].astype(int), 0, h - 1)
    d = depth_m[ys, xs]
    d = d[np.isfinite(d) & (d > 0.5) & (d < 120.0)]
    if d.size < MIN_MATCHES // 2:
        return None
    return float(np.median(d))


# A ground feature must be at least this far ahead to be usable for scale. Very close road texture is
# where the flat-plane assumption and the mount height are least reliable, and where a pixel of feature
# error is the largest fraction of the distance.
GROUND_MIN_M = 4.0
GROUND_MAX_M = 40.0
MIN_GROUND_FEATURES = 12


def scale_from_ground(pts_a: np.ndarray, pts_b: np.ndarray, calib, *, height_m: float) -> float | None:
    """Baseline in metres from how far road-surface features closed on the camera between two frames.

    The road is a plane a known height below the camera, so a pixel below the horizon has a metric
    distance. A static ground point ahead at distance Z is at Z minus the baseline one frame later, so
    the median closing distance over ground features is how far the vehicle travelled. This is the
    fallback when no metric depth exists for the frame, and it is weaker for the reasons it assumes: a
    flat road and a correct mount height.

    Median over features, and only forward closings inside a sane band, because half the features in this
    corpus sit on vehicles that are themselves moving and would otherwise set the scale.
    """
    from services.hdmap.georef import ipm_pixel_to_vehicle

    if len(pts_a) < MIN_GROUND_FEATURES:
        return None
    pitch = math.radians(float(calib.rpy_deg[1]))
    fisheye = getattr(calib, "model", "pinhole") == "fisheye"
    closings = []
    for (ua, va), (ub, vb) in zip(pts_a, pts_b, strict=False):
        ga = ipm_pixel_to_vehicle(float(ua), float(va), calib.fx, calib.fy, calib.cx, calib.cy,
                                  height_m, pitch, list(calib.dist or []), fisheye)
        gb = ipm_pixel_to_vehicle(float(ub), float(vb), calib.fx, calib.fy, calib.cx, calib.cy,
                                  height_m, pitch, list(calib.dist or []), fisheye)
        if ga is None or gb is None:
            continue
        za, zb = ga[0], gb[0]
        if not (GROUND_MIN_M <= za <= GROUND_MAX_M):
            continue
        closings.append(za - zb)
    if len(closings) < MIN_GROUND_FEATURES:
        return None
    # The signed median, returned as a magnitude. A forward camera sees ground points close on it and a
    # rear one sees them recede, so a filter that kept only closings recovered scale on 3 of 1,032 pairs
    # of a rear-facing rig session while working fine on a dashcam. The median still rejects the moving
    # vehicles that make up half the features in this corpus, because they are outliers in either sign.
    return abs(float(np.median(closings)))


def _integrate(steps: list[dict]) -> list[dict]:
    """Turn per-step rotations and metric baselines into a session-local ENU trajectory.

    A step with no metric baseline still turns the heading: the rotation is recoverable without scale and
    throwing it away would lose real information. The position simply does not advance across it, and the
    row records `speed_mps` null so nothing reads the stall as the vehicle having stopped.
    """
    x = y = yaw = 0.0
    out = []
    for st in steps:
        yaw += st["dyaw"]
        dist = st.get("dist")
        if dist is not None:
            # The camera's forward direction in the ENU frame after turning.
            x += dist * math.cos(yaw)
            y += dist * math.sin(yaw)
        qw, qx, qy, qz = yaw_to_quat(yaw)
        out.append({**st, "x": x, "y": y, "z": 0.0, "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                    "yaw": yaw})
    return out


async def _gnss_rows(db: AsyncSession, session_id: uuid.UUID) -> list[dict]:
    """Poses from GNSS fixes, or an empty list. These are the only measured rows this module can write."""
    from services.intelligence.egostate import derive_ego_state

    # `Frame.gnss` is a PostGIS geography point, not JSON. Selecting the column itself hands back a
    # geoalchemy element whose attribute access raises, so latitude and longitude are extracted in the
    # database with ST_Y and ST_X, which is how the other five readers of this column already do it.
    geom = cast(Frame.gnss, Geometry)
    rows = (await db.execute(
        select(Frame.frame_id, Frame.ts_ns, func.ST_Y(geom), func.ST_X(geom), Frame.ego_speed)
        .where(Frame.session_id == session_id, Frame.gnss.isnot(None))
        .order_by(Frame.ts_ns))).all()
    fixes = []
    for fid, ts, lat, lon, speed in rows:
        if lat is None or lon is None:
            continue
        fixes.append((fid, int(ts), float(lat), float(lon),
                      float(speed) if speed is not None else None))
    if len(fixes) < 2:
        return []
    lat0, lon0 = fixes[0][2], fixes[0][3]
    derived = derive_ego_state([(ts, lat, lon, sp) for _f, ts, lat, lon, sp in fixes])
    out = []
    for (fid, ts, lat, lon, _sp), d in zip(fixes, derived, strict=False):
        east, north = enu_offset(lat0, lon0, lat, lon)
        heading = d.get("heading")
        qw, qx, qy, qz = yaw_to_quat(float(heading) if heading is not None else 0.0)
        out.append({"frame_id": fid, "ts_ns": ts, "x": east, "y": north, "z": 0.0,
                    "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                    "speed_mps": d.get("speed"), "yaw_rate": d.get("yaw_rate"),
                    "source": GNSS, "quality": 0.9, "measured": True})
    return out


async def _depth_by_ts(db: AsyncSession, session_id: uuid.UUID) -> dict[int, str]:
    """Timestamps of this session's pseudo-LiDAR clouds. Their depth is the metric scale reference."""
    rows = (await db.execute(
        select(PointCloud.ts_ns, PointCloud.cloud_uri)
        .where(PointCloud.session_id == session_id, PointCloud.source == "pseudo"))).all()
    return {int(ts): uri for ts, uri in rows}


def _depth_from_cloud(cloud_uri: str, width: int, height: int) -> np.ndarray | None:
    """Recover a sparse depth image from a stored cloud, or None when it cannot be read.

    The cloud is already the metric depth this session paid a GPU to compute; re-running the depth model
    to recover a number it has already stored would be the expensive way to learn the same fact.
    """
    try:
        from services.lidar.ingest.store import load_cloud

        cloud = load_cloud(cloud_uri)
    except Exception:  # noqa: BLE001 - an unreadable cloud means no scale, not a failed session
        return None
    xyz = getattr(cloud, "xyz", None)
    if xyz is None or len(xyz) == 0:
        return None
    # Ego frame is x forward, y left, z up. Forward distance is the depth the camera saw.
    fwd = np.asarray(xyz[:, 0], dtype=np.float32)
    good = fwd[np.isfinite(fwd) & (fwd > 0.5) & (fwd < 120.0)]
    if good.size < 100:
        return None
    # A full reprojection would need the per-point pixel index, which the stored cloud does not keep.
    # The median forward distance of the whole cloud is the scale reference instead, returned as a
    # constant image so the caller's sampling code is the same in both paths.
    return np.full((height, width), float(np.median(good)), dtype=np.float32)


def _feature_mask(vehicle_id: str, cam_id: str, shape: tuple[int, int]) -> np.ndarray | None:
    """Everything except the car's own bonnet, or None when this camera has no hood mask.

    The bonnet is rigidly attached to the camera, so its features never move between frames no matter
    what the vehicle does. They are not merely useless to visual odometry: they are a block of perfectly
    stationary correspondences that pull the essential matrix toward "no motion" and crowd real ground
    features out of a fixed ORB budget. The same per-camera mask the detector cleanup sweep estimates
    answers this, so no new estimation is needed.
    """
    from services.autolabel.ego_mask import get_ego_mask

    ego = get_ego_mask(vehicle_id, cam_id)
    if ego is None or ego.area_frac <= 0:
        return None
    grid = np.asarray(ego.grid, dtype=np.uint8)
    hood = cv2.resize(grid, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(hood > 0, 0, 255).astype(np.uint8)


def _pick_camera(rows: list, cams: set[str]) -> str:
    """The camera to run odometry on: the one that covers the most of the session's timeline.

    One camera only, because stitching two cameras' motion into one trajectory double-counts every step.

    Frame count, and not a preference for a forward-facing camera, because the corpus contradicted that
    preference. On the first real rig session here, `rear_wide` gives 1,033 poses with 709 recovered and
    42 metrically scaled, while `front_narrow` gives 179 poses with 73 recovered and 0 scaled. The
    ground-plane scale reads how far road features moved, so what it needs is a wide view of road
    surface, and a narrow forward lens sees less of it than a wide rear one. Which way the camera points
    stopped mattering once the scale became a magnitude rather than a closing distance.
    """
    counts = {c: sum(1 for r in rows if r[2] == c) for c in cams}
    return max(cams, key=lambda c: counts.get(c, 0))


def _camera_height(calib) -> float:
    """The camera's height above the road, from the session's calibration or the configured rig default.

    A calibrated mount z is preferred because it was measured for this vehicle. The config default is
    the nominal forward-camera height and is what the IPM has always used, so falling back to it puts
    this scale on exactly the footing every other ground-plane computation here already stands on.
    """
    from core.config import get_settings

    z = float(getattr(calib, "xyz_m", (0.0, 0.0, 0.0))[2] or 0.0)
    if z > 0.3:
        return z
    return float(get_settings().rig.camera_height_m)


def _decode(img_uri: str) -> np.ndarray | None:
    from core.storage import get_object_store

    try:
        buf = np.frombuffer(get_object_store().get_bytes(img_uri), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception:  # noqa: BLE001 - a missing blob skips one frame, never the session
        return None


async def _visual_rows(db: AsyncSession, session_id: uuid.UUID, *, cam_id: str | None,
                       limit: int | None, vehicle_id: str) -> tuple[list[dict], dict]:
    """Poses from monocular visual odometry over one camera's frames, with the scale source recorded."""
    from services.calibration.resolve import resolve_calibration

    q = (select(Frame.frame_id, Frame.ts_ns, Frame.cam_id, Frame.img_uri, Frame.width, Frame.height)
         .where(Frame.session_id == session_id, Frame.img_uri.isnot(None))
         .order_by(Frame.ts_ns))
    rows = (await db.execute(q)).all()
    if not rows:
        return [], {"reason": "the session has no frames with images"}
    cam = cam_id or _pick_camera(rows, {r[2] for r in rows})
    rows = [r for r in rows if r[2] == cam]
    if limit:
        rows = rows[:limit]
    if len(rows) < 2:
        return [], {"reason": f"camera {cam} has fewer than two frames"}

    w = int(rows[0][4] or 1280)
    h = int(rows[0][5] or 960)
    calib = await resolve_calibration(session_id, cam, w, h)
    K = calib.K()
    depth_ts = await _depth_by_ts(db, session_id)

    orb = _detector()
    mask = _feature_mask(vehicle_id, cam, (h, w))
    stats_mask = mask is not None
    steps: list[dict] = []
    prev = None
    height_m = _camera_height(calib)
    stats = {"cam": cam, "pairs": 0, "recovered": 0, "scaled_by_depth": 0, "scaled_by_ground": 0,
             "unscaled": 0, "skipped_gap": 0, "unreadable": 0, "calibration_source": calib.source,
             "camera_height_m": height_m}
    for fid, ts, _c, uri, _w, _h in rows:
        img = _decode(uri)
        if img is None:
            stats["unreadable"] += 1
            prev = None
            continue
        kp, desc = orb.detectAndCompute(img, mask)
        if prev is None:
            prev = (fid, int(ts), kp, desc, img.shape)
            steps.append({"frame_id": fid, "ts_ns": int(ts), "dyaw": 0.0, "dist": 0.0, "dt": None,
                          "quality": 0.2, "scale_source": "origin"})
            continue

        dt = (int(ts) - prev[1]) / 1e9
        stats["pairs"] += 1
        if dt <= 0 or dt > MAX_STEP_S:
            # A gap in the recording. Integrating across it would draw a trajectory the car never took,
            # so the chain restarts here and the row says the step is unknown.
            stats["skipped_gap"] += 1
            steps.append({"frame_id": fid, "ts_ns": int(ts), "dyaw": 0.0, "dist": None, "dt": dt,
                          "quality": 0.0, "scale_source": "gap"})
            prev = (fid, int(ts), kp, desc, img.shape)
            continue

        rel = relative_pose(prev[2], prev[3], kp, desc, K)
        if rel is None:
            steps.append({"frame_id": fid, "ts_ns": int(ts), "dyaw": 0.0, "dist": None, "dt": dt,
                          "quality": 0.0, "scale_source": "no_pose"})
            prev = (fid, int(ts), kp, desc, img.shape)
            continue
        stats["recovered"] += 1

        dist = None
        scale_source = "none"
        cloud_uri = depth_ts.get(prev[1])
        if cloud_uri:
            depth = _depth_from_cloud(cloud_uri, w, h)
            median_depth = scale_from_depth(depth, rel["pts_a"], rel["t_unit"])
            if median_depth is not None:
                # The unit translation's forward component times the scene's metric depth. Clamped by a
                # plausible ground speed: a baseline above it is a scale failure, not a fast car, and
                # writing it would put a spike in the trajectory that no smoothing removes.
                fwd = abs(float(rel["t_unit"][2]))
                cand = fwd * median_depth
                if 0.0 <= cand <= MAX_SPEED_MPS * dt:
                    dist = cand
                    scale_source = "pseudo_depth"
                    stats["scaled_by_depth"] += 1
        if dist is None:
            # No metric depth here. The road plane is the other metric reference this rig has: the
            # camera height is known, so a ground feature's distance is known, and how far those
            # features closed is how far the vehicle moved.
            cand = scale_from_ground(rel["pts_a"], rel["pts_b"], calib, height_m=height_m)
            if cand is not None and 0.0 <= cand <= MAX_SPEED_MPS * dt:
                dist = cand
                scale_source = "ground_plane"
                stats["scaled_by_ground"] += 1
        if dist is None:
            stats["unscaled"] += 1

        quality = min(0.75, rel["inliers"] / float(ORB_FEATURES)) if dist is not None else 0.15
        steps.append({"frame_id": fid, "ts_ns": int(ts), "dyaw": rel["yaw"], "dist": dist, "dt": dt,
                      "quality": round(float(quality), 3), "scale_source": scale_source,
                      "inliers": rel["inliers"]})
        prev = (fid, int(ts), kp, desc, img.shape)

    stats["hood_masked"] = stats_mask
    placed = _integrate(steps)
    out = []
    for st in placed:
        dt = st.get("dt")
        speed = (st["dist"] / dt) if (st.get("dist") is not None and dt) else None
        yaw_rate = (st["dyaw"] / dt) if dt else None
        out.append({"frame_id": st["frame_id"], "ts_ns": st["ts_ns"], "x": st["x"], "y": st["y"],
                    "z": st["z"], "qw": st["qw"], "qx": st["qx"], "qy": st["qy"], "qz": st["qz"],
                    "speed_mps": speed, "yaw_rate": yaw_rate, "source": VISUAL,
                    "quality": st["quality"], "measured": False})
    return out, stats


async def build_ego_pose(session_id: uuid.UUID, *, cam_id: str | None = None,
                         limit: int | None = None, run_id: uuid.UUID | None = None,
                         replace: bool = True) -> dict:
    """Write this session's ego trajectory from the best source it has, and say which one that was.

    GNSS wins where it exists because it is the only source here that observed position. Visual odometry
    fills the rest and is written with `measured=False`, never blended into the GNSS rows: a fused
    trajectory would need a filter whose covariances nothing in this corpus can supply, and averaging a
    measurement with an inference produces a number that is neither.
    """
    from db.session import get_sessionmaker

    maker = get_sessionmaker()
    async with maker() as db:
        sess = await db.get(DbSession, session_id)
        if sess is None:
            return {"error": "session not found", "session_id": str(session_id)}
        gnss = await _gnss_rows(db, session_id)
        rows, stats = (gnss, {"source": GNSS}) if gnss else await _visual_rows(
            db, session_id, cam_id=cam_id, limit=limit, vehicle_id=sess.vehicle_id)
        if not rows:
            return {"session_id": str(session_id), "poses": 0, "measured": 0,
                    "reason": stats.get("reason", "no pose could be recovered from this session")}

        if replace:
            await db.execute(delete(EgoPose).where(EgoPose.session_id == session_id))
            await db.commit()

        written = 0
        for i in range(0, len(rows), POSE_BATCH):
            for r in rows[i:i + POSE_BATCH]:
                db.add(EgoPose(session_id=session_id, run_id=run_id, **r))
                written += 1
            await db.commit()

    measured = sum(1 for r in rows if r["measured"])
    report = {"session_id": str(session_id), "poses": written, "measured": measured,
              "source": rows[0]["source"], **{k: v for k, v in stats.items() if k != "reason"}}
    log.info("ego_pose.built", **{k: v for k, v in report.items() if not isinstance(v, dict)})
    return report


async def pose_at(db: AsyncSession, session_id: uuid.UUID, ts_ns: int) -> EgoPose | None:
    """The pose recorded at this exact timestamp, or None. No interpolation: a pose between two rows is
    a guess, and the callers here want to know when they are guessing."""
    return (await db.execute(select(EgoPose).where(
        EgoPose.session_id == session_id, EgoPose.ts_ns == ts_ns))).scalar_one_or_none()
