"""Camera-LiDAR alignment: score how well a LiDAR scan lines up with a camera image, and search for the
extrinsic that lines them up best (tool #9, /calibration/align).

Three measurements, because each sees a different failure:

- **Edge alignment** (Levinson & Thrun, 2013). Where the range jumps between neighbouring LiDAR returns there is
  an object boundary, and an object boundary is usually an image edge. Each scan point gets a weight from the
  range discontinuity on either side of it in its own range image (computed once per scan, independent of the
  extrinsic), the image gets an edge map softened by an inverse distance transform, and the score is the
  weighted mean of the edge map under the projected edge points. 1.0 means every depth edge lands on an image
  edge; a misaligned extrinsic drops it fast.
- **Mutual information** (Pandey et al., 2012) between LiDAR reflectivity and image brightness at the projected
  pixels. Paint, plates and signs are bright in both. Normalised so it reads 0..1.
- **Reprojection error** from point pairs a person clicked: a LiDAR point and the pixel where it really is. This
  is the only one with units of pixels, and it reuses `lidar_camera.reprojection_error` so the number agrees
  with the validation report.

The refinement is a bounded coordinate search over roll, pitch, yaw, x, y, z, coarse to fine, maximising the
edge score plus the information score summed over several scans. It is local by design: it polishes a
calibration that is roughly right (the dataset's, or a person's drag) and cannot rescue one that is wildly off,
because a far-off extrinsic projects the scan onto unrelated edges that score just as well.

The projection here is the resolver's own: cam = (ego - xyz) @ R(rpy), pinhole, done in NumPy float64 on purpose.
`services/lidar/project.py` sends large clouds to the GPU, and a search that projects thousands of times in the
API process must not take the card.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import cv2
import numpy as np

# Range-image resolution for the discontinuity weights: 0.18 degrees of azimuth is the HDL-64's own horizontal
# step at 10 Hz, and 64 rows match its beams. Other sensors are binned the same way; a coarser sensor leaves
# empty cells, which simply have no neighbour.
AZ_BINS = 2000
EL_BINS = 64
# A jump smaller than this between neighbouring returns is texture, foliage or noise, not a silhouette. Measured on
# KITTI 0015 (three frames, true calibration against a wrong extrinsic that scores almost as well): at 0.3 m the
# edge score preferred the wrong extrinsic by 0.002; at 1.0 m the true one won by 0.013, at 2.0 m by 0.023. Leaves
# and hedges make many small jumps whose image edges are everywhere, so they flatten the score.
# A sparse scene (few objects, or a narrow-field sensor) may have too few such silhouettes to score on; the threshold
# then steps down until at least MIN_EDGE_POINTS remain.
MIN_JUMP_M = 2.0
JUMP_LADDER_M = (2.0, 1.0, 0.5, 0.3)
MIN_EDGE_POINTS = 600
# The image edge map is softened over this many pixels, so a point two pixels off an edge still scores.
EDGE_SIGMA_PX = 3.0
MI_BINS = 32

# The search: how far from the start it may go, and the step ladder. A local polish, see the module note.
MAX_ROT_DEG = 3.0
MAX_TRANS_M = 0.3
ROT_STEPS_DEG = (0.8, 0.4, 0.2, 0.1, 0.05)
TRANS_STEPS_M = (0.08, 0.04, 0.02, 0.01, 0.005)
MI_WEIGHT = 1.0
# The rotation grid: its step, and the edge-field softness it is scored against (the step moves a point ~6 px
# at KITTI's focal length, so the field must still have a slope half a step away).
GRID_STEP_DEG = 0.5
GRID_SIGMA_PX = 6.0
# Non-boundary points kept per scan for the information term during the grid (the MI histogram's sample).
GRID_MI_POINTS = 4000
# Score units given up per metre the translation moves from the start. Translation is weakly observable (see
# `refine`), so without a prior it drifts to whatever the fit frames' noise prefers.
TRANS_PRIOR = 0.2


def with_extrinsics(cal, rpy_deg, xyz_m):
    """A copy of a resolved Calibration with a different extrinsic (intrinsics untouched)."""
    return dataclasses.replace(cal, rpy_deg=tuple(float(v) for v in rpy_deg), xyz_m=tuple(float(v) for v in xyz_m))


def project(xyz: np.ndarray, cal, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(uv, in_image, depth) for ego-frame points under `cal`, pinhole, float64, CPU."""
    cam = (np.asarray(xyz, dtype=np.float64) - np.asarray(cal.xyz_m, dtype=np.float64)) @ cal.R().astype(np.float64)
    z = cam[:, 2]
    front = z > 1e-3
    safe = np.where(front, z, 1.0)
    uv = np.stack([cam[:, 0] / safe * cal.fx + cal.cx, cam[:, 1] / safe * cal.fy + cal.cy], 1)
    inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h)
    return uv, inside, z


def edge_weights(xyz: np.ndarray, min_jump_m: float | None = None) -> np.ndarray:
    """Per-point depth-discontinuity weight from the scan's own range image (0 = not on a boundary).

    A point is weighted by how much further its horizontal neighbours are, so the near side of a boundary (the
    object's silhouette, which is what the camera draws as an edge) carries the weight, not the background.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    rng = np.linalg.norm(xyz, axis=1)
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    if n > 100:
        # A spinning LiDAR stored in firing order (KITTI's .bin, kept by the importer) has each point's horizontal
        # neighbours beside it in the array, which is exact where the binned range image below is not: the
        # HDL-64's beams are not evenly spaced in elevation, so equal-width rows mix two beams and invent jumps.
        step = np.abs(np.diff(az))
        step = np.minimum(step, 2 * math.pi - step)
        near = step < math.radians(1.0)
        if near.mean() > 0.9:
            prev = np.r_[np.inf, np.where(near, rng[:-1], np.inf)]
            nxt = np.r_[np.where(near, rng[1:], np.inf), np.inf]
            jump = np.maximum(np.where(np.isfinite(prev), prev - rng, 0.0), np.where(np.isfinite(nxt), nxt - rng, 0.0))
            return _threshold(jump, min_jump_m)
    el = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
    col = np.clip(((az + math.pi) / (2 * math.pi) * AZ_BINS).astype(np.int64), 0, AZ_BINS - 1)
    lo, hi = float(el.min()), float(el.max())
    row = np.clip(((el - lo) / max(hi - lo, 1e-6) * (EL_BINS - 1) + 0.5).astype(np.int64), 0, EL_BINS - 1)
    cell = row * AZ_BINS + col
    # The nearest return per cell, so a cell's range is the surface the beam actually hit first.
    img = np.full(EL_BINS * AZ_BINS, np.inf)
    np.minimum.at(img, cell, rng)
    img = img.reshape(EL_BINS, AZ_BINS)
    left = img[row, (col - 1) % AZ_BINS]
    right = img[row, (col + 1) % AZ_BINS]
    jump = np.maximum(np.where(np.isfinite(left), left - rng, 0.0), np.where(np.isfinite(right), right - rng, 0.0))
    return _threshold(jump, min_jump_m)


def _threshold(jump: np.ndarray, min_jump_m: float | None) -> np.ndarray:
    """sqrt(jump) weights above the threshold; with none given, the highest rung of the ladder that keeps
    MIN_EDGE_POINTS."""
    if min_jump_m is None:
        min_jump_m = JUMP_LADDER_M[-1]
        for t in JUMP_LADDER_M:
            if int((jump >= t).sum()) >= MIN_EDGE_POINTS:
                min_jump_m = t
                break
    w = np.sqrt(np.clip(jump, 0.0, 9.0))
    w[jump < min_jump_m] = 0.0
    return w.astype(np.float32)


def edge_distance(gray: np.ndarray) -> np.ndarray:
    """Distance in pixels from every pixel to the nearest image edge (Canny, thresholds from the median)."""
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 1.2)
    med = float(np.median(g))
    edges = cv2.Canny(g, max(10, int(0.66 * med)), max(30, int(1.33 * med)))
    return cv2.distanceTransform(np.where(edges > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)


def edge_map(gray: np.ndarray, sigma_px: float = EDGE_SIGMA_PX) -> np.ndarray:
    """The image's edges as a 0..1 field: 1 on an edge, falling off as exp(-distance / sigma)."""
    return np.exp(-edge_distance(gray) / sigma_px).astype(np.float32)


def sample(field: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-pixel lookup of a 2D field at in-image uv."""
    h, w = field.shape[:2]
    u = np.clip(uv[:, 0].astype(np.int64), 0, w - 1)
    v = np.clip(uv[:, 1].astype(np.int64), 0, h - 1)
    return field[v, u]


def edge_score(uv: np.ndarray, inside: np.ndarray, weights: np.ndarray, emap: np.ndarray) -> float | None:
    """Weighted mean of the edge field under the projected boundary points, or None if none are in view."""
    m = inside & (weights > 0)
    if not np.any(m):
        return None
    w = weights[m]
    return float((sample(emap, uv[m]) * w).sum() / w.sum())


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = MI_BINS) -> float | None:
    """Normalised mutual information of two samples in 0..1: I(a;b) / min(H(a), H(b))."""
    if len(a) < 50:
        return None
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist / hist.sum()
    px, py = pxy.sum(1), pxy.sum(0)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    denom = min(hx, hy)
    return mi / denom if denom > 1e-9 else None


@dataclass
class AlignView:
    """One scan and its camera frame, prepared once so a score is a projection and two lookups."""

    xyz: np.ndarray            # (N, 3) ego frame, already restricted to points that can face the camera
    intensity: np.ndarray      # (N,) 0..1
    weights: np.ndarray        # (N,) edge weights
    emap: np.ndarray           # (H, W) edge field at EDGE_SIGMA_PX
    gray: np.ndarray           # (H, W) uint8
    width: int
    height: int
    frame_id: str | None = None
    dist: np.ndarray | None = None   # (H, W) distance to the nearest edge, for wider fields at coarse levels

    def at_sigma(self, sigma_px: float) -> AlignView:
        """The same view scored against a softer edge field (a wider basin for the coarse search levels)."""
        if self.dist is None or sigma_px == EDGE_SIGMA_PX:
            return self
        return dataclasses.replace(self, emap=np.exp(-self.dist / sigma_px).astype(np.float32))


def prepare_view(xyz: np.ndarray, intensity: np.ndarray, image_bgr: np.ndarray, *, frame_id: str | None = None,
                 max_points: int = 60000, seed: int = 0) -> AlignView:
    """Precompute the extrinsic-independent parts: edge weights over the whole scan (they need its neighbours),
    then a bounded subset that keeps every boundary point and a random sample of the rest for the MI term."""
    xyz = np.asarray(xyz, dtype=np.float32)
    inten = np.asarray(intensity, dtype=np.float32) if intensity is not None else np.zeros(len(xyz), np.float32)
    w = edge_weights(xyz)
    gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    keep = np.flatnonzero(w > 0)
    rest = np.flatnonzero(w <= 0)
    budget = max(0, max_points - len(keep))
    if len(rest) > budget:
        rest = np.random.default_rng(seed).choice(rest, budget, replace=False)
    idx = np.sort(np.concatenate([keep, rest]))
    h, wd = gray.shape[:2]
    dist = edge_distance(gray)
    return AlignView(xyz=xyz[idx], intensity=inten[idx], weights=w[idx],
                     emap=np.exp(-dist / EDGE_SIGMA_PX).astype(np.float32), gray=gray,
                     width=int(wd), height=int(h), frame_id=frame_id, dist=dist)


def score_view(view: AlignView, cal, with_mi: bool = True) -> dict:
    uv, inside, _ = project(view.xyz, cal, view.width, view.height)
    e = edge_score(uv, inside, view.weights, view.emap)
    m = inside
    mi = (mutual_information(view.intensity[m], sample(view.gray, uv[m]).astype(np.float32))
          if with_mi and m.sum() else None)
    return {"edge": e, "mi": mi, "in_image": int(inside.sum()),
            "edge_points": int((inside & (view.weights > 0)).sum())}


def score(views: list[AlignView], cal, mi_weight: float = MI_WEIGHT) -> dict:
    """The mean edge and MI scores over several scans, and the objective the refinement maximises."""
    per = [score_view(v, cal, with_mi=mi_weight > 0) for v in views]
    edges = [p["edge"] for p in per if p["edge"] is not None]
    mis = [p["mi"] for p in per if p["mi"] is not None]
    edge = float(np.mean(edges)) if edges else None
    mi = float(np.mean(mis)) if mis else None
    objective = (edge or 0.0) + mi_weight * (mi or 0.0)
    return {"edge": None if edge is None else round(edge, 5), "mi": None if mi is None else round(mi, 5),
            "objective": round(objective, 5), "in_image": int(sum(p["in_image"] for p in per)),
            "edge_points": int(sum(p["edge_points"] for p in per)), "views": len(views)}


def edge_only(view: AlignView, keep_other: int = 0, seed: int = 1) -> AlignView:
    """The view restricted to its boundary points (all the edge score reads), plus `keep_other` of the rest for
    the information term: the whole score at a fraction of the cost, for the grid stage."""
    m = view.weights > 0
    if keep_other > 0:
        rest = np.flatnonzero(~m)
        if len(rest) > keep_other:
            rest = np.random.default_rng(seed).choice(rest, keep_other, replace=False)
        m = m.copy()
        m[rest] = True
    return dataclasses.replace(view, xyz=view.xyz[m], intensity=view.intensity[m], weights=view.weights[m])


def refine(views: list[AlignView], cal0, *, max_rot_deg: float = MAX_ROT_DEG, max_trans_m: float = MAX_TRANS_M,
           grid_step_deg: float = GRID_STEP_DEG, grid_sigma_px: float = GRID_SIGMA_PX,
           rot_steps=ROT_STEPS_DEG, trans_steps=TRANS_STEPS_M, mi_weight: float = MI_WEIGHT,
           trans_grid: bool = False, trans_prior: float = TRANS_PRIOR, on_level=None) -> dict:
    """Search for the extrinsic maximising `score(...).objective`, bounded around `cal0`.

    Three stages, because the six parameters are not equally observable and the score has local maxima:

    1. Rotation on a grid over the whole bound (against a softened edge field, on a subsample). A few
       centimetres of translation moves a point 30 m away by well under a pixel, so rotation is what puts the scan
       on the image, and a coordinate search started from a rotated extrinsic stops on the first set of unrelated
       edges it finds (measured on KITTI 0015: from a 25 px start it ended 66 px off). The grid sees every basin.
    2. Rotation, coordinate search against the sharp edge field from the best grid point.
    3. All six together, edge plus mutual information, coarse to fine on the step ladder, with a mild prior
       (`trans_prior`) holding the translation near the start. A translation grid (`trans_grid`) was measured
       and is off by default: on both the KITTI drive and the ray-cast street it traded rotation for
       translation and ended further from the truth (KITTI 0015 from a 25 px start: 7.0 px with it, 4.1 px
       without).

    Never returns something that scores worse than the start. The score reported is the raw score, without the
    prior.
    """
    p0 = np.array(list(cal0.rpy_deg) + list(cal0.xyz_m), dtype=np.float64)
    lo = p0 - np.array([max_rot_deg] * 3 + [max_trans_m] * 3)
    hi = p0 + np.array([max_rot_deg] * 3 + [max_trans_m] * 3)
    evals = [0]

    def obj(p: np.ndarray, vs: list[AlignView], w_mi: float) -> float:
        evals[0] += 1
        prior = trans_prior * float(np.linalg.norm(p[3:] - p0[3:]))
        return score(vs, with_extrinsics(cal0, p[:3], p[3:]), w_mi)["objective"] - prior

    def search(p: np.ndarray, vs: list[AlignView], steps: np.ndarray, dims: range, w_mi: float):
        best_p, best = p.copy(), obj(p, vs, w_mi)
        improved, rounds = True, 0
        while improved and rounds < 12:
            improved, rounds = False, rounds + 1
            for k in dims:
                for sgn in (1.0, -1.0):
                    cand = best_p.copy()
                    cand[k] = float(np.clip(cand[k] + sgn * steps[k], lo[k], hi[k]))
                    if cand[k] == best_p[k]:
                        continue
                    v = obj(cand, vs, w_mi)
                    if v > best + 1e-6:
                        best_p, best, improved = cand, v, True
        return best_p, best, rounds

    def tick(i: int, best: float) -> None:
        if on_level is not None:
            on_level(i, 2 + len(rot_steps), best)

    before = score(views, cal0, mi_weight)
    edges = [edge_only(v, keep_other=GRID_MI_POINTS) for v in views]
    soft = [v.at_sigma(grid_sigma_px) for v in edges]
    # Stage 1: block grids, rotation then translation then a finer rotation, each over its whole bound with the
    # other block held. Translation gets a grid too because a start that is off by 10 cm moves near points by tens
    # of pixels, and a rotation grid alone then picks the rotation that best hides that.
    def grid(center: np.ndarray, block: slice, half: float, step: float, cur: float) -> tuple[np.ndarray, float]:
        n = int(round(half / step))
        offs = np.arange(-n, n + 1) * step
        best_c, best_v = center, cur
        for a in offs:
            for b in offs:
                for c in offs:
                    cand = center.copy()
                    cand[block] = np.clip(center[block] + np.array([a, b, c]), lo[block], hi[block])
                    v = obj(cand, soft, mi_weight)
                    if v > best_v + 1e-9:
                        best_c, best_v = cand, v
        return best_c, best_v

    best_p, best = p0.copy(), obj(p0, soft, mi_weight)
    best_p, best = grid(best_p, slice(0, 3), max_rot_deg, grid_step_deg, best)
    if trans_grid:
        best_p, best = grid(best_p, slice(3, 6), max_trans_m, max_trans_m / 6, best)
    best_p, best = grid(best_p, slice(0, 3), 2 * grid_step_deg, grid_step_deg / 2, best)
    levels = [{"stage": "block grids", "step_deg": grid_step_deg, "sigma_px": grid_sigma_px,
               "objective": round(best, 5)}]
    tick(1, best)
    # Stage 2: rotation, sharp field.
    for rs in (grid_step_deg / 2, grid_step_deg / 5):
        best_p, best, _r = search(best_p, edges, np.array([rs] * 3 + [0.0] * 3), range(3), mi_weight)
    levels.append({"stage": "rotation", "objective": round(best, 5)})
    tick(2, best)
    # Stage 3: everything, with the information term.
    for li, (rs, ts) in enumerate(zip(rot_steps, trans_steps, strict=True)):
        best_p, best, rounds = search(best_p, views, np.array([rs] * 3 + [ts] * 3), range(6), mi_weight)
        levels.append({"stage": "joint", "rot_step_deg": rs, "trans_step_m": ts, "objective": round(best, 5),
                       "rounds": rounds})
        tick(3 + li, best)
    after = score(views, with_extrinsics(cal0, best_p[:3], best_p[3:]), mi_weight)
    if after["objective"] <= before["objective"] + 1e-6:
        best_p, after = p0, before
    return {"rpy_deg": [round(float(v), 5) for v in best_p[:3]], "xyz_m": [round(float(v), 5) for v in best_p[3:]],
            "before": before, "after": after, "improved": after["objective"] > before["objective"] + 1e-6,
            "evaluations": evals[0], "levels": levels,
            "at_bound": bool(np.any(np.isclose(best_p, lo)) or np.any(np.isclose(best_p, hi)))}


def projection_shift(xyz: np.ndarray, cal_a, cal_b, img_w: int, img_h: int) -> dict:
    """How far the scan's projection moves between two calibrations, in pixels (points in view under both).

    Against a reference calibration this is a reprojection error over every point of the scan, which is how the
    refinement is checked on a scene whose true calibration is known.
    """
    uva, ina, _ = project(xyz, cal_a, img_w, img_h)
    uvb, inb, _ = project(xyz, cal_b, img_w, img_h)
    m = ina & inb
    if not np.any(m):
        return {"mean_px": None, "median_px": None, "max_px": None, "n": 0}
    d = np.linalg.norm(uva[m] - uvb[m], axis=1)
    return {"mean_px": round(float(d.mean()), 3), "median_px": round(float(np.median(d)), 3),
            "max_px": round(float(d.max()), 3), "n": int(m.sum())}


def solve_from_pairs(points_ego: np.ndarray | list, uv: np.ndarray | list, cal0) -> dict:
    """The extrinsic that best explains clicked pairs (PnP from the current guess), as resolver rpy/xyz.

    Needs at least four pairs; six or more spread across the image is what makes it trustworthy.
    """
    from services.lidar.calib.kitti import extrinsics_from_affine

    pts = np.asarray(points_ego, dtype=np.float64).reshape(-1, 3)
    obs = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 4:
        raise ValueError("at least four point pairs are needed to solve the extrinsic")
    r0 = cal0.R().astype(np.float64).T                     # cam_col = R0 @ ego - R0 @ t
    t0 = -r0 @ np.asarray(cal0.xyz_m, dtype=np.float64)
    rvec0, _ = cv2.Rodrigues(r0)
    ok, rvec, tvec = cv2.solvePnP(pts, obs, cal0.K(), np.zeros(5), rvec0.copy(), t0.reshape(3, 1).copy(),
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("the pairs do not determine a camera pose")
    rot, _ = cv2.Rodrigues(rvec)
    rpy, xyz = extrinsics_from_affine(rot, tvec.reshape(3))
    return {"rpy_deg": rpy, "xyz_m": xyz}


def pairs_error(points_ego, uv, cal, img_w: int, img_h: int) -> dict:
    """RMS / max pixel error of clicked pairs under `cal`, through the validation module's own function."""
    from services.lidar.calib.lidar_camera import reprojection_error

    if len(points_ego) == 0:
        return {"rms": None, "max": None, "n": 0}
    return reprojection_error(points_ego, uv, cal.cam_id, img_w, img_h, cal)
