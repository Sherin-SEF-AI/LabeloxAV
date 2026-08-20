"""Moving a box to the next frame by moving the camera, not by assuming the world stood still.

Label propagation copies a box from one frame to the next and lets a human correct it. The copy assumes
the object did not move relative to the camera, which on a moving vehicle is wrong for everything: a
parked car sweeps across the image as the ego passes it, and a sign at the roadside sweeps fastest of all.
So propagation is worst exactly on the static objects it should be best at, and an annotator spends their
time dragging boxes that geometry could have placed.

For anything ON THE GROUND PLANE the correction is exact, and it does not need the object's motion. A
ground point seen by a calibrated camera has one world position; move the camera by a known rigid
transform and the point projects somewhere computable. The map from image to image is a homography induced
by the ground plane:

    H = K @ (R - (t @ n^T) / d) @ K^-1

with n the ground normal in the SOURCE camera frame, d the camera height above it, and (R, t) the pose
change: R the rotation and t the translation of the camera CENTRE from the source pose to the destination
pose, expressed in source camera coordinates. Driving forward is therefore t = [0, 0, +z], and a box on
the road grows and descends in the image, which is what an annotator sees. No per-pixel depth is needed,
which is the entire reason to use the ground plane rather than a structure-from-motion step.

That sign is stated because it is the easy thing to get backwards: the same formula with t as the motion
of the SCENE relative to the camera produces a homography that is wrong in exactly the way that looks
plausible, shrinking an approaching object instead of growing it. `test_driving_forward_moves_a_ground_box_down_and_outward`
pins the direction rather than the formula.

THE ASSUMPTION IS NAMED AND CHECKED. This is exact only for points on the plane. A box's bottom edge sits
on the ground for a vehicle, a pedestrian and a cone; it does not for a gantry, an overhead sign or a
traffic light, and warping those by the ground homography moves them the wrong way and confidently. So the
caller passes a motion model per class (packs/base.py::ClassTree's sibling surface, `motion_model`), and
this refuses rather than guessing for anything not on the ground.

WHAT IT REFUSES. No calibration, no ego motion, a degenerate homography, or a warped box that leaves the
image: each returns a result with `measured` False and a reason, never a silently wrong box. A propagated
box a human has to notice is wrong is worse than no box, because the box is evidence and its absence is
not.

NumPy is the oracle; the torch path mirrors the batch warp and must agree to 1e-6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

_EPS = 1e-9


@dataclass(frozen=True)
class WarpedBox:
    """One propagated box, or a refusal with the reason.

    `box` is xyxy in the destination frame. `shrink` is the area ratio to the source box, reported because
    a homography that shrinks a box to a tenth of itself has almost certainly been applied to something
    that was not on the ground.
    """

    measured: bool
    box: tuple[float, float, float, float] | None
    shrink: float | None
    reason: str | None = None


def ground_homography(K: npt.ArrayLike, R: npt.ArrayLike, t: npt.ArrayLike,
                      normal: npt.ArrayLike, height: float) -> npt.NDArray[np.float64] | None:
    """The image-to-image homography induced by a plane, or None when it is degenerate.

    `R` and `t` are the camera's pose change from the source frame to the destination frame, with t the
    translation of the camera centre expressed in SOURCE camera coordinates: driving forward is
    t = [0, 0, +z]. `normal` is the plane normal in the source camera frame and `height` the camera's
    distance to it.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    n = np.asarray(normal, dtype=np.float64).reshape(3, 1)
    if height <= _EPS:
        return None
    nn = float(np.linalg.norm(n))
    if nn < _EPS:
        return None
    n = n / nn
    try:
        Kinv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        return None
    H = K @ (R - (t @ n.T) / float(height)) @ Kinv
    if not np.isfinite(H).all() or abs(float(np.linalg.det(H))) < 1e-12:
        return None
    return H / H[2, 2] if abs(H[2, 2]) > _EPS else H


def _warp_np(H: npt.NDArray[np.float64], pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Project (N, 2) points through H. Points behind the camera come back as NaN rather than mirrored.

    A negative homogeneous w means the point mapped behind the image plane; dividing anyway produces a
    plausible-looking coordinate on the wrong side, which is the worst possible failure for a box a human
    is about to trust.
    """
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    out = hom @ H.T
    w = out[:, 2:3]
    bad = np.abs(w) < _EPS
    res = np.divide(out[:, :2], np.where(bad, 1.0, w))
    res[bad[:, 0] | (w[:, 0] < 0)] = np.nan
    return res


def _warp_torch(H: npt.NDArray[np.float64], pts: npt.NDArray[np.float64],
                device: str) -> npt.NDArray[np.float64]:  # pragma: no cover - GPU path
    h = torch.as_tensor(H, device=device, dtype=torch.float64)
    p = torch.as_tensor(pts, device=device, dtype=torch.float64)
    hom = torch.cat([p, torch.ones((p.shape[0], 1), device=device, dtype=torch.float64)], dim=1)
    out = hom @ h.T
    w = out[:, 2:3]
    bad = (w.abs() < _EPS) | (w < 0)
    res = out[:, :2] / torch.where(w.abs() < _EPS, torch.ones_like(w), w)
    res = torch.where(bad, torch.full_like(res, float("nan")), res)
    arr: npt.NDArray[np.float64] = res.cpu().numpy()
    return arr


def warp_box(box: npt.ArrayLike, H: npt.ArrayLike | None, *, width: int, height: int,
             motion_model: str = "moving", max_shrink: float = 0.1,
             device: str | None = None) -> WarpedBox:
    """Move one xyxy box through the ground homography, or refuse and say why.

    `motion_model` comes from the pack. Only "static_ground" is warped: a static ELEVATED object (a gantry,
    an overhead sign) is not on the plane and the ground homography moves it confidently in the wrong
    direction, and a MOVING object's displacement is not a function of the ego motion at all.
    """
    if motion_model != "static_ground":
        return WarpedBox(False, None, None,
                         f"motion model '{motion_model}' is not on the ground plane, so a ground "
                         "homography does not describe how it moves between frames")
    if H is None:
        return WarpedBox(False, None, None, "no usable ground homography (missing or degenerate)")

    b = np.asarray(box, dtype=np.float64).reshape(4)
    Hm = np.asarray(H, dtype=np.float64).reshape(3, 3)
    corners = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], dtype=np.float64)

    if _HAS_TORCH and device is not None and str(device) != "cpu" and torch.cuda.is_available():
        w = _warp_torch(Hm, corners, str(device))  # pragma: no cover - GPU path
    else:
        w = _warp_np(Hm, corners)
    if not np.isfinite(w).all():
        return WarpedBox(False, None, None, "the box maps behind the camera in the destination frame")

    x1, y1 = float(w[:, 0].min()), float(w[:, 1].min())
    x2, y2 = float(w[:, 0].max()), float(w[:, 1].max())
    if x2 <= x1 or y2 <= y1:
        return WarpedBox(False, None, None, "the warped box has no area")

    src_area = max((b[2] - b[0]) * (b[3] - b[1]), _EPS)
    shrink = ((x2 - x1) * (y2 - y1)) / src_area
    if shrink < max_shrink or shrink > 1.0 / max_shrink:
        # A ground homography does not change a box's area by an order of magnitude between consecutive
        # frames. It means the box was not on the plane, or the motion estimate is wrong.
        return WarpedBox(False, None, round(shrink, 6),
                         f"the warp changes the box area by {shrink:.2f}x, which a ground homography "
                         "between consecutive frames does not do")

    cx1, cy1 = max(0.0, x1), max(0.0, y1)
    cx2, cy2 = min(float(width), x2), min(float(height), y2)
    if cx2 - cx1 < 1.0 or cy2 - cy1 < 1.0:
        return WarpedBox(False, None, round(shrink, 6),
                         "the warped box leaves the destination frame")
    return WarpedBox(True, (round(cx1, 3), round(cy1, 3), round(cx2, 3), round(cy2, 3)),
                     round(shrink, 6))


def warp_boxes(boxes: npt.ArrayLike, H: npt.ArrayLike | None, motion_models: list[str], *,
               width: int, height: int, device: str | None = None) -> dict[str, Any]:
    """Warp many boxes. Returns {"warped", "n_warped", "n_refused", "reasons"}."""
    bs = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = [warp_box(bs[i], H, width=width, height=height,
                    motion_model=motion_models[i] if i < len(motion_models) else "moving",
                    device=device)
           for i in range(bs.shape[0])]
    reasons: dict[str, int] = {}
    for w in out:
        if not w.measured and w.reason:
            key = w.reason.split(",")[0][:60]
            reasons[key] = reasons.get(key, 0) + 1
    return {"warped": tuple(out),
            "n_warped": sum(1 for w in out if w.measured),
            "n_refused": sum(1 for w in out if not w.measured),
            "reasons": reasons}


__all__ = ["WarpedBox", "ground_homography", "warp_box", "warp_boxes"]
