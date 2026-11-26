"""A small ray-cast street for the driving-tool tests: one LiDAR scan and one camera image of the same scene.

The scene is ground (dark, with two bright lane stripes), three boxes (vehicles), two poles and a textured wall.
The LiDAR sits at the ego origin 1.73 m above the road (KITTI's Velodyne frame); the camera is rendered through a
known Calibration. Because both sensors see the same geometry, an alignment search started from a perturbed
extrinsic has a right answer to find, which a real drive cannot give a test.
"""

from __future__ import annotations

import math

import numpy as np

GROUND_Z = -1.73
BOXES = [  # (xmin, xmax, ymin, ymax, zmin, zmax, intensity)
    (12.0, 16.2, -3.6, -1.8, GROUND_Z, -0.2, 0.55),
    (20.0, 24.5, 1.9, 3.8, GROUND_Z, 0.1, 0.45),
    (9.0, 10.2, 4.5, 5.2, GROUND_Z, 0.3, 0.6),
    (17.0, 17.3, -6.3, -6.0, GROUND_Z, 3.5, 0.8),     # pole
    (26.0, 26.3, 5.0, 5.3, GROUND_Z, 3.5, 0.8),       # pole
]
WALL_X = 38.0


def _ground_intensity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    stripe = (np.abs(np.abs(y) - 1.7) < 0.12) & ((np.floor(x / 3.0) % 2) == 0)
    return np.where(stripe, 0.95, 0.12)


def cast(origin: np.ndarray, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest hit distance and surface intensity along each ray (inf / 0 where nothing is hit)."""
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(dirs, dtype=np.float64)
    n = len(d)
    best = np.full(n, np.inf)
    inten = np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = (GROUND_Z - o[2]) / d[:, 2]
        ok = (d[:, 2] < -1e-9) & (tg > 0)
        p = o + d * tg[:, None]
        better = ok & (tg < best)
        best = np.where(better, tg, best)
        inten = np.where(better, _ground_intensity(p[:, 0], p[:, 1]), inten)
        tw = (WALL_X - o[0]) / d[:, 0]
        ok = (d[:, 0] > 1e-9) & (tw > 0)
        p = o + d * tw[:, None]
        ok &= (p[:, 2] > GROUND_Z) & (p[:, 2] < 6.0)
        better = ok & (tw < best)
        tex = 0.25 + 0.2 * ((np.floor(p[:, 1] / 1.5) + np.floor(p[:, 2] / 1.0)) % 2)
        best = np.where(better, tw, best)
        inten = np.where(better, tex, inten)
        for x0, x1, y0, y1, z0, z1, val in BOXES:
            lo, hi = np.array([x0, y0, z0]), np.array([x1, y1, z1])
            t1 = (lo - o) / d
            t2 = (hi - o) / d
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            ok = (tmax >= tmin) & (tmin > 0)
            better = ok & (tmin < best)
            best = np.where(better, tmin, best)
            inten = np.where(better, val, inten)
    return best, inten


def lidar_scan(az_deg=(-70.0, 70.0, 0.18), el_deg=(-24.0, 2.0, 64)) -> tuple[np.ndarray, np.ndarray]:
    az = np.radians(np.arange(*az_deg))
    el = np.radians(np.linspace(el_deg[0], el_deg[1], el_deg[2]))
    a, e = np.meshgrid(az, el)
    dirs = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1).reshape(-1, 3)
    t, inten = cast(np.zeros(3), dirs)
    ok = np.isfinite(t) & (t < 80)
    return (dirs[ok] * t[ok, None]).astype(np.float32), inten[ok].astype(np.float32)


def render(cal, width: int = 1242, height: int = 375) -> np.ndarray:
    """A grey image of the scene through `cal` (cam = (ego - xyz) @ R)."""
    v, u = np.mgrid[0:height, 0:width].astype(np.float64)
    dcam = np.stack([(u - cal.cx) / cal.fx, (v - cal.cy) / cal.fy, np.ones_like(u)], -1).reshape(-1, 3)
    dego = dcam @ cal.R().astype(np.float64).T
    dego /= np.linalg.norm(dego, axis=1, keepdims=True)
    t, inten = cast(np.asarray(cal.xyz_m, dtype=np.float64), dego)
    shade = np.where(np.isfinite(t), 30 + 200 * inten, 235.0)
    img = shade.reshape(height, width)
    rng = np.random.default_rng(3)
    img = img + rng.normal(0, 2.0, img.shape)
    g = np.clip(img, 0, 255).astype(np.uint8)
    return np.dstack([g, g, g])


def true_calibration():
    from services.calibration.resolve import Calibration

    return Calibration(cam_id="cam_front", model="pinhole", fx=721.5, fy=721.5, cx=609.6, cy=172.9,
                       rpy_deg=(0.3, 0.6, -0.4), xyz_m=(0.27, -0.06, -0.08), source="dataset", quality=0.95)


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
