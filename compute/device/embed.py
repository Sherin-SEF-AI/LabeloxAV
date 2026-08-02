"""A frame descriptor a rig can actually afford, and the honest cost of using it.

The server decides novelty on DINOv3, which is the right descriptor and needs a GPU and half a gigabyte of
weights. Some rigs have that. A dashcam-class board does not, and requiring it would mean the selection
never runs anywhere it matters most, since the fleets with the tightest uplink budgets are exactly the ones
with the smallest boards.

So there are two descriptors and the choice is explicit rather than automatic-with-a-fallback. Automatic
would mean a fleet silently splitting into devices that select well and devices that select badly, with
nothing in the uploaded data saying which was which, and the resulting corpus would have a sampling bias
nobody could later characterise. `describe()` takes the backend by name and the manifest records it.

**What the cheap descriptor costs.** A tiled intensity-and-gradient histogram measures whether the picture
changed, not whether the scene did. It cannot tell a pedestrian stepping off a kerb from a cloud moving, and
it will call a headlight flare a scene change. Against DINOv3 on this corpus that means more frames kept for
a given novelty threshold, and the frames it keeps are less interesting. It is a floor, not a substitute,
and it is far better than fixed-rate sampling, which is the actual alternative on a board with no GPU.
"""

from __future__ import annotations

import numpy as np

BACKENDS = ("tiled_histogram", "dinov3")

# 4x4 tiles over the frame. Enough spatial structure that a car entering on the left is not cancelled out by
# one leaving on the right, which a whole-frame histogram cannot see; coarse enough to survive the camera
# shake and rolling shutter of a vehicle mount.
TILES = 4
BINS = 8


def describe(image_bgr: np.ndarray, *, backend: str = "tiled_histogram") -> np.ndarray:
    """A unit-norm descriptor for one frame.

    Raises on an unknown backend rather than falling back, so a typo in a device config is caught at the
    first frame instead of producing a fleet that samples differently from the rest and never says so.
    """
    if backend == "tiled_histogram":
        return tiled_histogram(image_bgr)
    if backend == "dinov3":
        return _dinov3(image_bgr)
    raise ValueError(f"unknown descriptor backend '{backend}'; known: {BACKENDS}")


def tiled_histogram(image_bgr: np.ndarray) -> np.ndarray:
    """Contrast-normalised intensity and gradient histograms per tile, concatenated and normalised.

    Two halves, and both are needed. Gradient energy answers "is there structure here" and is naturally
    steady under exposure change. Intensity answers "what does it look like", and raw intensity is defeated
    by the thing that happens constantly on a road: the sun comes out, every bin shifts together, and a
    static scene reads as novel.

    So intensity is histogrammed after normalising against the whole frame's mean and spread. A first
    version histogrammed raw intensity, and a test comparing the same scene 25 grey levels brighter against
    a completely different scene preferred the different scene, 0.91 to 0.998: that descriptor would have
    spent a device's entire uplink budget on the weather.

    Normalised globally rather than per tile, which was the second attempt. Per-tile normalisation makes
    every uniform tile identical whatever its brightness, so a white truck filling the left of the frame and
    a black one filling the right become the same descriptor. Against the frame statistics an exposure shift
    still cancels, because it moves every tile and the mean together, while the difference between a bright
    tile and a dark one survives.
    """
    img = np.asarray(image_bgr)
    if img.ndim == 3:
        # Rec. 601 luma. Cheaper than a colour conversion call and adequate for a change detector.
        gray = (0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    h, w = gray.shape[:2]
    if h < TILES or w < TILES:
        return _unit(np.histogram(gray, bins=BINS, range=(0, 255))[0].astype(np.float32))

    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    g_mean, g_std = float(gray.mean()), float(gray.std())

    feats = []
    ys = np.linspace(0, h, TILES + 1).astype(int)
    xs = np.linspace(0, w, TILES + 1).astype(int)
    for i in range(TILES):
        for j in range(TILES):
            tile = gray[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            tmag = mag[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            # Against the frame's own statistics, so a uniform brightness or gain change moves nothing
            # while a tile genuinely brighter than its neighbours still reads that way. Ranged over +/-3
            # standard deviations, which covers the distribution without one specular highlight setting the
            # scale.
            norm = (tile - g_mean) / (g_std + 1e-6)
            feats.append(np.histogram(norm, bins=BINS, range=(-3.0, 3.0))[0].astype(np.float32))
            # The gradient range is data-dependent, so it is capped rather than scaled by the tile's own
            # maximum: per-tile normalisation would make a flat wall and a busy junction look alike.
            feats.append(np.histogram(tmag, bins=BINS, range=(0.0, 128.0))[0].astype(np.float32))
    return _unit(np.concatenate(feats))


def _dinov3(image_bgr: np.ndarray) -> np.ndarray:
    """The server's descriptor, for a rig that can carry it.

    Imported lazily so that importing this module on a board without torch does not fail. A device that asks
    for this backend and cannot run it should fail loudly at startup, which it will.
    """
    from core.embeddings import dinov3

    return _unit(np.asarray(dinov3().encode_image(image_bgr), dtype=np.float32))


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v if n < 1e-9 else v / n
