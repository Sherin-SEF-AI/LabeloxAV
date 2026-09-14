"""Counterfactual perturbations of real frames: does the model still see it when the world changes a little.

Gold measures a model on the conditions gold happens to contain. An Indian road at dusk in the rain with a
truck half-blocking a rider is not a rare condition, it is Tuesday, and a model can score well on a sealed
gold set while failing on all of it because the gold set was collected in daylight.

A perturbation is a controlled change to a real frame: the same scene, one thing different. Recall
measured before and after is a causal statement about that one thing, which is what neither a gold score
nor a slice metric can give. A slice says the model is worse on dark frames; it cannot say whether that is
the darkness or the fact that dark frames in this corpus are also mostly highways.

**Every function here is deterministic under its seed.** A counterfactual whose result moves between runs
cannot be compared to the previous run, so the finding would be unreproducible in exactly the situation
where somebody wants to check it.

**Perturbed frames are never given objects.** A perturbation moves pixels and does not move the truth: the
rider behind the added occlusion is still a rider at the same box. Writing labels for a perturbed frame
would create a second copy of every object with a different origin, and the whole measurement is that the
labels stay fixed while the pixels change.
"""

from __future__ import annotations

import numpy as np

from core.logging import get_logger

log = get_logger("perturb")

# The named perturbations, and the strength each is measured at by default. Named because the gate refuses
# on specific ones and a set that drifted between the gate and the evaluator would refuse on nothing.
DEFAULT_STRENGTHS = {
    "occlude": 0.3,
    "dusk": 0.45,
    "rain": 0.5,
    "fog": 0.5,
    "motion_blur": 0.5,
}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def occlude(image: np.ndarray, box: list[float], frac: float = 0.3, *, seed: int = 7) -> np.ndarray:
    """Cover `frac` of an object's box with an opaque patch, from a randomly chosen edge.

    From an edge rather than the centre, because that is how occlusion actually happens: a vehicle passes
    in front, a pole crosses the view, a bystander steps in. A centred hole leaves the object's outline
    intact on all four sides, which is a much easier problem than the one being asked about.
    """
    out = image.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = (int(max(0, min(v, lim))) for v, lim in
                      zip(box, (w, h, w, h), strict=False))
    if x1 <= x0 or y1 <= y0 or frac <= 0:
        return out
    frac = float(min(1.0, frac))
    side = int(_rng(seed).integers(0, 4))
    bw, bh = x1 - x0, y1 - y0
    if side == 0:      # from the left
        out[y0:y1, x0:x0 + max(1, int(bw * frac))] = 0
    elif side == 1:    # from the right
        out[y0:y1, x1 - max(1, int(bw * frac)):x1] = 0
    elif side == 2:    # from the top
        out[y0:y0 + max(1, int(bh * frac)), x0:x1] = 0
    else:              # from the bottom, which is what a passing vehicle does
        out[y1 - max(1, int(bh * frac)):y1, x0:x1] = 0
    return out


def dusk(image: np.ndarray, gamma: float = 0.45) -> np.ndarray:
    """Darken as light falls, by gamma rather than by subtraction.

    Subtracting a constant clips the shadows to black and leaves the highlights linear, which is not what
    a camera does at dusk and is a much cruder change than the one being measured. Gamma compresses the
    whole range the way falling light does, so what is lost is contrast in the dark parts, which is
    exactly where a rider in dark clothing lives.
    """
    g = float(max(0.05, min(1.0, gamma)))
    lut = (np.linspace(0, 1, 256) ** (1.0 / g) * 255.0).astype(np.uint8)
    import cv2

    return cv2.LUT(image, lut)


def motion_blur(image: np.ndarray, strength: float = 0.5, *, seed: int = 7) -> np.ndarray:
    """Directional blur, as from camera shake or a fast pan on a rough road."""
    import cv2

    k = max(3, int(round(3 + strength * 18)) | 1)
    angle = float(_rng(seed).uniform(-25.0, 25.0))
    kern = np.zeros((k, k), dtype=np.float32)
    kern[k // 2, :] = 1.0 / k
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kern = cv2.warpAffine(kern, m, (k, k))
    total = kern.sum()
    if total > 0:
        kern /= total
    return cv2.filter2D(image, -1, kern)


def rain(image: np.ndarray, strength: float = 0.5, *, seed: int = 7) -> np.ndarray:
    """Streaks plus the veiling haze rain puts on a windscreen.

    Both halves matter. Streaks alone are salt-and-pepper noise a convolution shrugs off; the haze is what
    actually costs a detector its contrast, and a rain perturbation without it under-states the effect.
    """
    import cv2

    out = image.astype(np.float32)
    h, w = out.shape[:2]
    rng = _rng(seed)
    n = int(400 + strength * 2600)
    length = int(8 + strength * 22)
    layer = np.zeros((h, w), dtype=np.float32)
    xs = rng.integers(0, w, size=n)
    ys = rng.integers(0, max(1, h - length), size=n)
    slant = int(rng.integers(-4, 5))
    for x, y in zip(xs, ys, strict=False):
        cv2.line(layer, (int(x), int(y)), (int(x + slant), int(y + length)), 1.0, 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    out = out + layer[..., None] * (140.0 * strength)
    # The veil: a grey wash that lifts the blacks, which is what kills contrast in real rain.
    out = out * (1.0 - 0.25 * strength) + 255.0 * 0.18 * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def fog(image: np.ndarray, strength: float = 0.5, depth_m: np.ndarray | None = None) -> np.ndarray:
    """Atmospheric scattering, thicker with distance when a depth map is available.

    Fog is depth-dependent and a flat wash is not fog: the near car stays sharp while the far one
    disappears, and that difference is the whole reason fog breaks a detector at range and not up close.
    Without a depth map this falls back to a vertical gradient, which is a weaker approximation and is
    documented as one rather than presented as the same measurement.
    """
    out = image.astype(np.float32)
    h, w = out.shape[:2]
    if depth_m is not None and depth_m.shape[:2] == (h, w):
        d = np.clip(np.nan_to_num(depth_m.astype(np.float32), nan=80.0), 0.5, 120.0)
        # Beer-Lambert: transmission falls exponentially with distance.
        beta = 0.02 + 0.06 * float(strength)
        t = np.exp(-beta * d)[..., None]
        kind = "depth"
    else:
        # Distance grows upward toward the horizon in a forward frame, so transmission is highest at
        # the bottom (the near road) and falls toward the top. Written the other way round first, which
        # put the haze on the bonnet and left the horizon clear: fog that thins with distance. A stand-in
        # for a depth map, and named as one.
        grad = np.linspace(0.15, 1.0, h, dtype=np.float32)[:, None]
        t = (grad ** (0.5 + strength))[..., None]
        kind = "vertical_gradient"
    airlight = 235.0
    out = out * t + airlight * (1.0 - t)
    log.debug("perturb.fog", kind=kind)
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_object(image: np.ndarray, mask: np.ndarray, *, radius: int = 5) -> np.ndarray:
    """Inpaint an object away, for asking whether the model reports what is no longer there.

    The counterfactual nobody else can run: a detector that still fires on the removed object is keying on
    context rather than on the object, and no amount of measurement on unmodified frames reveals that.
    """
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if m.shape[:2] != image.shape[:2]:
        m = cv2.resize(m, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.inpaint(image, m, radius, cv2.INPAINT_TELEA)


def apply(name: str, image: np.ndarray, *, strength: float | None = None, box: list[float] | None = None,
          depth_m: np.ndarray | None = None, seed: int = 7) -> np.ndarray:
    """Apply a named perturbation at its default strength unless told otherwise."""
    s = float(strength if strength is not None else DEFAULT_STRENGTHS.get(name, 0.5))
    if name == "occlude":
        if box is None:
            raise ValueError("occlude needs the box of the object it is occluding")
        return occlude(image, box, frac=s, seed=seed)
    if name == "dusk":
        return dusk(image, gamma=s)
    if name == "rain":
        return rain(image, strength=s, seed=seed)
    if name == "fog":
        return fog(image, strength=s, depth_m=depth_m)
    if name == "motion_blur":
        return motion_blur(image, strength=s, seed=seed)
    raise ValueError(f"unknown perturbation {name!r}; known: {sorted(DEFAULT_STRENGTHS)}")
