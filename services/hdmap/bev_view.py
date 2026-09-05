"""A bird's-eye view of the road, and the two-way mapping that makes it drawable rather than only viewable.

Lane annotation here has always been image space, and for a good reason: that is where the paint is. It is
also where a lane is hardest to judge. In a forward camera the road runs to a vanishing point, so the far
half of a lane occupies a handful of pixels, two parallel lanes converge, and a curve and a lane change
look the same. Flattening the road removes all three: parallel lanes are parallel, a constant-width lane
has constant width, and a curve is a curve.

The reason this did not exist is that half the mathematics was missing. `ipm_pixel_to_vehicle` has lifted a
pixel to the ground since the georeferencing work; the inverse, a ground point back to the pixel that sees
it, appeared nowhere in the repo, and without it a bird's-eye view can be looked at and not drawn on. That
inverse is `services/hdmap/georef.py::vehicle_to_ipm_pixel` and it round-trips against the forward
direction to floating-point exactness.

**What this view can and cannot support.** The warp assumes a flat road, which is the same assumption the
depth and cuboid work already make and the same one that fails on a crest or a dip. And a pixel near the
horizon is worth many metres: at the far end of the range the warp is interpolating a few source pixels
across a lot of output, so a line drawn there is drawn on very little evidence. The far bound comes from
`ipm_max_range_m`, the codebase's own limit on where a flat-road lift stops meaning anything, rather than
from a number picked to make the picture look good.
"""

from __future__ import annotations

import math

import numpy as np

from core.logging import get_logger

log = get_logger("bev_view")

# The patch of road the view covers, in metres. Lateral is symmetric about the camera axis.
DEFAULT_NEAR_M = 2.0
DEFAULT_HALF_WIDTH_M = 12.0

# Output resolution. At 20 px/m a lane line is about 2 px wide, which is enough to aim at and cheap enough
# to warp on every frame open.
DEFAULT_PX_PER_M = 20.0

# The view is capped so an annotator is not offered a canvas whose far half is invented. Beyond the range
# a flat-road lift supports, one source pixel spreads over many output pixels and the warp is drawing
# interpolation rather than road.
MAX_FAR_M = 60.0
MIN_FAR_M = 8.0


def ipm_range_m(fy: float, height_m: float) -> float:
    """How far this camera's flat-road lift can be believed, from the same derivation dynamics uses."""
    from services.dynamics.compute import BOX_BOTTOM_JITTER_PX, IPM_ERROR_BUDGET

    return IPM_ERROR_BUDGET * fy * max(0.1, height_m) / BOX_BOTTOM_JITTER_PX


class BevView:
    """The metric patch of road a BEV raster covers, and how to convert between the two.

    Held as an object rather than a pile of floats because every consumer needs the same four conversions
    and getting one of them subtly wrong produces a picture that looks right and is off by a metre.
    """

    def __init__(self, near_m: float, far_m: float, half_width_m: float, px_per_m: float):
        self.near_m = float(near_m)
        self.far_m = float(far_m)
        self.half_width_m = float(half_width_m)
        self.px_per_m = float(px_per_m)
        self.width = max(1, int(round(2 * self.half_width_m * self.px_per_m)))
        self.height = max(1, int(round((self.far_m - self.near_m) * self.px_per_m)))

    def to_metric(self, bx: float, by: float) -> tuple[float, float]:
        """A BEV pixel to (forward, lateral) metres. Row 0 is the FAR edge, so the view reads like a map
        with the camera at the bottom."""
        forward = self.far_m - (float(by) / self.px_per_m)
        lateral = (float(bx) / self.px_per_m) - self.half_width_m
        return forward, lateral

    def to_pixel(self, forward: float, lateral: float) -> tuple[float, float]:
        """(forward, lateral) metres to a BEV pixel. The exact inverse of `to_metric`."""
        by = (self.far_m - float(forward)) * self.px_per_m
        bx = (float(lateral) + self.half_width_m) * self.px_per_m
        return bx, by

    def as_dict(self) -> dict:
        return {"near_m": round(self.near_m, 2), "far_m": round(self.far_m, 2),
                "half_width_m": round(self.half_width_m, 2), "px_per_m": round(self.px_per_m, 3),
                "width": self.width, "height": self.height}


def view_for(cal, *, near_m: float = DEFAULT_NEAR_M, half_width_m: float = DEFAULT_HALF_WIDTH_M,
             px_per_m: float = DEFAULT_PX_PER_M, far_m: float | None = None) -> BevView:
    """The patch of road worth showing for one camera, bounded by what its lift can support."""
    height_m = abs(float(cal.xyz_m[2])) or 1.5
    limit = ipm_range_m(float(cal.fy), height_m)
    far = far_m if far_m is not None else limit
    far = max(MIN_FAR_M, min(MAX_FAR_M, far, limit))
    return BevView(near_m, far, half_width_m, px_per_m)


def _cal_args(cal) -> dict:
    """The arguments the georef IPM pair takes, from a resolved calibration."""
    return {
        "fx": float(cal.fx), "fy": float(cal.fy), "cx": float(cal.cx), "cy": float(cal.cy),
        # The mount height is the camera's z above the road; pitch is the mount's downward tilt.
        "height_m": abs(float(cal.xyz_m[2])) or 1.5,
        "pitch_rad": math.radians(float(cal.rpy_deg[1])),
        "dist": list(cal.dist or []),
        "fisheye": (cal.model == "fisheye"),
    }


def build_maps(cal, view: BevView) -> tuple[np.ndarray, np.ndarray]:
    """Per-BEV-pixel source coordinates, for `cv2.remap`.

    Computed with the same `vehicle_to_ipm_pixel` the interactive path uses rather than with a separate
    homography, so what an annotator draws on lands exactly where the warp said it would. A homography
    would be faster and would be a second implementation of the same projection, free to disagree with the
    first on a fisheye camera, where the mapping is not a homography at all.
    """
    from services.hdmap.georef import vehicle_to_ipm_pixel

    args = _cal_args(cal)
    ys, xs = np.mgrid[0:view.height, 0:view.width]
    forward = view.far_m - (ys.astype(np.float64) / view.px_per_m)
    lateral = (xs.astype(np.float64) / view.px_per_m) - view.half_width_m

    map_x = np.full(forward.shape, -1.0, dtype=np.float32)
    map_y = np.full(forward.shape, -1.0, dtype=np.float32)

    if not args["fisheye"]:
        # The pinhole case is a closed form over the whole grid, so it is done at once rather than per
        # pixel: a 480x740 view is 355,200 calls otherwise.
        x, y, z = lateral, np.full_like(lateral, args["height_m"]), forward
        if args["pitch_rad"]:
            c, sn = math.cos(-args["pitch_rad"]), math.sin(-args["pitch_rad"])
            y, z = y * c - z * sn, y * sn + z * c
        ok = z > 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            u = args["fx"] * x / z + args["cx"]
            v = args["fy"] * y / z + args["cy"]
        map_x = np.where(ok, u, -1.0).astype(np.float32)
        map_y = np.where(ok, v, -1.0).astype(np.float32)
        return map_x, map_y

    for r in range(view.height):
        for c in range(view.width):
            uv = vehicle_to_ipm_pixel(float(forward[r, c]), float(lateral[r, c]), **args)
            if uv is None:
                continue
            map_x[r, c], map_y[r, c] = uv
    return map_x, map_y


def render(img: np.ndarray, cal, view: BevView) -> np.ndarray:
    """Warp a camera frame into the bird's-eye view."""
    import cv2

    map_x, map_y = build_maps(cal, view)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def bev_to_image(points: list[list[float]], cal, view: BevView) -> list[list[float]]:
    """BEV pixels back to image pixels: what a line drawn on the warp means on the frame.

    This is the half that did not exist. Points that fall behind the camera are dropped rather than
    clamped, because a clamped point is a coordinate nobody drew that would be stored as though they had.
    """
    from services.hdmap.georef import vehicle_to_ipm_pixel

    args = _cal_args(cal)
    out = []
    for p in points:
        if len(p) < 2:
            continue
        forward, lateral = view.to_metric(float(p[0]), float(p[1]))
        uv = vehicle_to_ipm_pixel(forward, lateral, **args)
        if uv is None:
            continue
        out.append([round(uv[0], 2), round(uv[1], 2)])
    return out


def image_to_bev(points: list[list[float]], cal, view: BevView) -> list[list[float]]:
    """Image pixels to BEV pixels: where an existing lane sits on the warp.

    A point above the horizon has no ground intersection and is dropped, which is the honest answer: a
    lane control point up there was never on the road plane to begin with.
    """
    from services.hdmap.georef import ipm_pixel_to_vehicle

    args = _cal_args(cal)
    out = []
    for p in points:
        if len(p) < 2:
            continue
        g = ipm_pixel_to_vehicle(float(p[0]), float(p[1]), **args)
        if g is None:
            continue
        bx, by = view.to_pixel(g[0], g[1])
        out.append([round(bx, 2), round(by, 2)])
    return out
