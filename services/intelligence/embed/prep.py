"""Crop preprocessing for embeddings. A tight object crop is rarely square, but the encoders resize to a
square input: SigLIP 2's processor squishes (distorting aspect), and DINOv3's timm transform resizes the
short side then centre-crops (clipping the top/bottom of a tall pedestrian or the ends of a bus). Both lose
discriminative shape. Letterboxing the crop to a square first -- centre the object, pad the rest with a
neutral grey -- preserves aspect and full content, so the vector describes the object as it actually is.

Tiny crops (a distant VRU a few pixels tall) carry almost no signal after the model's own downsampling, so
they are upscaled to a floor first; the vector then reflects real structure instead of interpolation noise.
"""

from __future__ import annotations

import numpy as np

# Recorded on every vector's model_versions so a re-embed after a prep change is detectable in provenance.
PREP_TAG = "letterbox-v1"


def square_letterbox(bgr: np.ndarray, *, min_side: int = 48, pad: int = 114) -> np.ndarray:
    """Return a square, aspect-preserving crop: the object centred, the remainder padded with mid-grey.
    Upscales crops whose short side is below min_side so small objects are not degenerate."""
    import cv2

    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    h, w = bgr.shape[:2]
    m = min(h, w)
    if 0 < m < min_side:
        sc = min_side / m
        bgr = cv2.resize(bgr, (max(1, int(round(w * sc))), max(1, int(round(h * sc)))), interpolation=cv2.INTER_CUBIC)
        h, w = bgr.shape[:2]
    if h == w:
        return bgr
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    bottom, right = s - h - top, s - w - left
    return cv2.copyMakeBorder(bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad, pad, pad))
