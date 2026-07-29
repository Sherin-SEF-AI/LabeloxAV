"""Lane polylines from a lane-marking segmentation mask. CLRerNet is the configured pod lane model but is
blocked on the pod's modern torch (mmcv/mmdet have no wheel for torch 2.11/cu128). This derives lanes from
the lane-marking classes the Mapillary Mask2Former segmenter already produces: cluster the marking pixels
into individual lines (connected components), then sample each line into a control-point polyline in the
Lane.control_points shape. Pure over a binary mask, so it is tested without a model or a pod.

Connected components cannot represent a dashed lane, which is the problem `group_collinear` exists to fix. A
dashed line is not one component, it is one component per dash, so a single lane came out either as several
short stubs that each claimed to be a lane in their own right, or, when the dashes were shorter than the
minimum height, as nothing at all. Both are worse than they sound now that lane type is measured rather than
assumed: a stub is short enough to be entirely paint, so it reads as a confident *solid* line, and a solid
line is one that crossing is an offence. Fragmenting a dashed lane manufactures violations.

So the components are grouped back into lanes before anything sees them. Dashes of one lane are collinear and
the gaps between them are regular, which is the same evidence the line-type classifier uses further down the
pipe, applied here to decide what is one lane rather than what kind of lane it is.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.logging import get_logger

log = get_logger("lane_marking")


def lanes_from_marking_mask(mask, min_pixels: int = 80, min_height: int = 20, n_points: int = 8,
                            max_lanes: int = 8) -> list[list[list[float]]]:
    """mask: HxW bool/uint8 of lane-marking pixels. Returns up to max_lanes polylines [[x, y], ...], one per
    marking line, ordered longest first. A line must have >= min_pixels and span >= min_height rows to count
    (filters specks and short dashes' noise; a real lane line is tall and thin)."""
    m = np.asarray(mask).astype(np.uint8)
    if m.ndim == 3:
        m = m[..., 0]
    n_labels, labels = cv2.connectedComponents(m)
    lanes: list[list[list[float]]] = []
    for lab in range(1, n_labels):
        ys, xs = np.where(labels == lab)
        if len(xs) < min_pixels:
            continue
        y0, y1 = int(ys.min()), int(ys.max())
        if y1 - y0 < min_height:
            continue
        band = max(2.0, (y1 - y0) / (2.0 * n_points))
        pts: list[list[float]] = []
        for yl in np.linspace(y0, y1, n_points):
            sel = np.abs(ys - yl) <= band
            if sel.any():
                pts.append([round(float(xs[sel].mean()), 1), round(float(yl), 1)])
        if len(pts) >= 2:
            lanes.append(pts)
    lanes.sort(key=lambda p: -(p[-1][1] - p[0][1]))           # longest vertical span first
    return lanes[:max_lanes]


# How far a fragment's extrapolated position may sit from a candidate lane's, as a fraction of frame width,
# for the two to be the same lane. Dashes of one lane are collinear to within the noise of the segmenter;
# a neighbouring lane is a lane width away, which is far more than this.
COLLINEAR_TOL_FRAC = 0.02
# Fragments whose directions differ by more than this are not the same line, however close they pass. Two
# lanes converging toward the horizon cross without being one lane.
MAX_ANGLE_DIFF_DEG = 12.0


def _fit_line(poly: list[list[float]]) -> tuple[float, float, float, float] | None:
    """x as a function of y for one fragment: slope, intercept, and the y range it covers.

    A function of y rather than of x because a lane is near-vertical in an image, where a fit of y on x is
    ill-conditioned exactly when the lane is best behaved.
    """
    pts = np.array([[float(p[0]), float(p[1])] for p in poly if p is not None and len(p) >= 2],
                   dtype=np.float64)
    if len(pts) < 2:
        return None
    y, x = pts[:, 1], pts[:, 0]
    if float(y.max() - y.min()) < 1e-6:
        return None
    slope, intercept = np.polyfit(y, x, 1)
    return float(slope), float(intercept), float(y.min()), float(y.max())


def _angle_deg(slope: float) -> float:
    return float(np.degrees(np.arctan(slope)))


def group_collinear(polys: list[list[list[float]]], frame_width: int,
                    tol_frac: float = COLLINEAR_TOL_FRAC) -> list[list[list[float]]]:
    """Merge fragments that lie along one line into single polylines.

    The dashes of a lane are collinear, so each fragment's fitted line predicts where the others sit. Two
    fragments join when each one's line predicts the other's midpoint to within the tolerance, which is a
    stricter test than one predicting the other and is what stops a short, badly fitted stub from annexing a
    lane it merely points at.

    Merging is transitive: a chain of dashes joins into one lane even though the first and last are far
    enough apart that neither would be matched to the other directly.
    """
    fits = [(_fit_line(p), p) for p in polys]
    usable = [(f, p) for f, p in fits if f is not None]
    if len(usable) < 2:
        return list(polys)

    tol = max(2.0, tol_frac * max(1, frame_width))
    n = len(usable)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        (si, bi, y0i, y1i), _ = usable[i]
        for j in range(i + 1, n):
            (sj, bj, y0j, y1j), _ = usable[j]
            if abs(_angle_deg(si) - _angle_deg(sj)) > MAX_ANGLE_DIFF_DEG:
                continue
            mid_i, mid_j = (y0i + y1i) / 2.0, (y0j + y1j) / 2.0
            # Each line has to predict the other's midpoint. One-way agreement is what a stub pointing at a
            # lane looks like, and it is not evidence they are the same line.
            if (abs((si * mid_j + bi) - (sj * mid_j + bj)) <= tol
                    and abs((sj * mid_i + bj) - (si * mid_i + bi)) <= tol):
                union(i, j)

    groups: dict[int, list[list[float]]] = {}
    for idx, (_fit, poly) in enumerate(usable):
        groups.setdefault(find(idx), []).extend([[float(p[0]), float(p[1])] for p in poly])

    out: list[list[list[float]]] = []
    for pts in groups.values():
        pts.sort(key=lambda p: p[1])
        out.append(pts)
    # Unfittable fragments are passed through rather than dropped: they are still marking pixels somebody
    # drew or a model found, and losing them silently is worse than carrying one short lane.
    out.extend([p for f, p in fits if f is None])
    out.sort(key=lambda p: -(p[-1][1] - p[0][1]))
    return out


def lanes_from_marking_mask_grouped(mask, frame_width: int | None = None, *, min_pixels: int = 20,
                                    min_height: int = 8, n_points: int = 8,
                                    max_lanes: int = 8) -> list[list[list[float]]]:
    """Lanes from a marking mask, with the dashes of one lane put back together.

    The thresholds are deliberately far below `lanes_from_marking_mask`'s. Those were sized for a whole lane,
    which is why a dash never survived them; here a fragment only has to be big enough to be a dash, because
    grouping is what turns fragments back into lanes. Filtering happens after grouping, on the merged lane,
    where a height threshold means what it was always supposed to mean.
    """
    m = np.asarray(mask)
    width = int(frame_width or (m.shape[1] if m.ndim >= 2 else 0)) or 1280
    frags = lanes_from_marking_mask(mask, min_pixels=min_pixels, min_height=min_height,
                                    n_points=n_points, max_lanes=10_000)
    merged = group_collinear(frags, width)
    kept = [p for p in merged if (p[-1][1] - p[0][1]) >= 20]
    log.info("lane_marking.grouped", fragments=len(frags), lanes=len(merged), kept=len(kept))
    return kept[:max_lanes]
