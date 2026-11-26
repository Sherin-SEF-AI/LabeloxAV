"""The no-training reconstruction: the window's LiDAR placed in one frame, coloured from the cameras, splatted.

This is the fallback when splats cannot be trained (no GPU, gsplat absent, office hours) and a check on them
when they can: it renders the same views from measured points only, so where the two disagree the splats
invented something. A point is drawn as a square whose side shrinks with depth, far to near so the nearest
point on a pixel wins; pixels no point reaches stay at the background colour rather than being filled in.
"""

from __future__ import annotations

import numpy as np


def render_points(xyz: np.ndarray, rgb: np.ndarray, T_world_cam: np.ndarray, K: np.ndarray, width: int,
                  height: int, *, ortho: bool = False, background=(0.6, 0.65, 0.7), base_px: float = 2.5,
                  ref_depth_m: float = 8.0, downscale: int = 2) -> np.ndarray:
    """Render coloured points into an HxWx3 float image in 0..1 (pinhole or orthographic camera).

    Drawn at 1/`downscale` resolution and resized up: a LiDAR scan's rings leave gaps between points a few
    pixels wide at full resolution, and closing them at the lower resolution costs nothing in honesty (no
    pixel is invented that no point reached, it is only drawn larger).
    """
    if downscale > 1 and width >= 64 and height >= 64:
        import cv2

        s = float(downscale)
        Ks = np.asarray(K, dtype=np.float64).copy()
        Ks[:2] /= s
        small = render_points(xyz, rgb, T_world_cam, Ks, max(1, int(round(width / s))),
                              max(1, int(round(height / s))), ortho=ortho, background=background,
                              base_px=base_px, ref_depth_m=ref_depth_m, downscale=1)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    img = np.empty((height, width, 3), dtype=np.float32)
    img[:] = np.asarray(background, dtype=np.float32)
    if len(xyz) == 0:
        return img
    T = np.linalg.inv(T_world_cam)
    cam = xyz @ T[:3, :3].T + T[:3, 3]
    z = cam[:, 2]
    front = z > 0.3
    if ortho:
        u = K[0, 0] * cam[:, 0] + K[0, 2]
        v = K[1, 1] * cam[:, 1] + K[1, 2]
        size = np.full(len(z), max(1, int(round(K[0, 0] * 0.12))), dtype=np.int64)
    else:
        zs = np.where(front, z, 1.0)
        u = K[0, 0] * cam[:, 0] / zs + K[0, 2]
        v = K[1, 1] * cam[:, 1] / zs + K[1, 2]
        size = np.clip(np.round(base_px * ref_depth_m / np.maximum(zs, 0.5)), 1, 9).astype(np.int64)
    ok = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return img
    order = idx[np.argsort(-z[idx], kind="stable")]           # far first, near last: the near one wins
    ui = u[order].astype(np.int64)
    vi = v[order].astype(np.int64)
    col = rgb[order].astype(np.float32)
    sz = size[order]
    for s in np.unique(sz):
        sel = sz == s
        half = int(s) // 2
        for dy in range(-half, int(s) - half):
            for dx in range(-half, int(s) - half):
                uu = np.clip(ui[sel] + dx, 0, width - 1)
                vv = np.clip(vi[sel] + dy, 0, height - 1)
                img[vv, uu] = col[sel]
    return img
