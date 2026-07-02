"""Interactive pixel-assist helpers: brush/eraser mask composition and SLIC superpixels. These run on
CPU and reuse the existing polygon utilities; the SAM point/box prompt (magic-wand) is served by the
existing sam_service, not here. Heavy imports (cv2, numpy, skimage) are inside the functions so the
module loads without them.
"""

from __future__ import annotations


def compose_mask(polygons: list[list[float]], ops: list[dict], width: int, height: int) -> list[list[float]]:
    """Rasterize the object's current polygons, apply circular brush (add) / eraser (subtract) stamps,
    and re-polygonize. ops is a list of {"op": "add"|"erase", "center": [x,y], "radius": r}.

    The rings are combined with the even-odd rule (XOR per ring), not a flat union, so a ring nested
    inside another reads as a hole. Without this the eraser looked broken: erasing the interior of a mask
    (for example an occluding vehicle out of a large wall region) punched a hole in the raster, but the
    old external-only re-polygonize discarded it and returned the unchanged outer boundary. Holes are now
    preserved on the way in (so successive strokes compound) and on the way out (keep_holes=True)."""
    import cv2
    import numpy as np

    from services.autolabel.paths.path_b_sam3 import polygons_from_mask

    m = np.zeros((int(height), int(width)), np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, np.float32).reshape(-1, 2).astype(np.int32)
        if len(pts) >= 3:
            ring = np.zeros_like(m)
            cv2.fillPoly(ring, [pts], 1)
            m ^= ring  # even-odd: a ring inside a filled region carves a hole
    for op in ops:
        cx, cy = int(round(op["center"][0])), int(round(op["center"][1]))
        cv2.circle(m, (cx, cy), max(1, int(round(op["radius"]))), 1 if op.get("op") == "add" else 0, -1)
    return polygons_from_mask(m.astype(bool), keep_holes=True)


def slic_superpixels(image_bgr, n_segments: int = 300, compactness: float = 12.0) -> list[list[float]]:
    """Oversegment the frame into SLIC superpixels, returned as the largest contour polygon per region,
    so an annotator can click a region to add it to a mask."""
    import cv2
    import numpy as np
    from skimage.segmentation import slic

    from services.autolabel.paths.path_b_sam3 import polygons_from_mask

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    seg = slic(rgb, n_segments=int(n_segments), compactness=float(compactness), start_label=0)
    out: list[list[float]] = []
    for lab in np.unique(seg):
        polys = polygons_from_mask(seg == lab)
        if polys:
            out.append(max(polys, key=len))  # the largest contour represents the superpixel footprint
    return out
