"""What kind of line is this, read off the paint the lane curve already points at.

`lane_type` has been the literal string "solid" in every path that writes a lane. Nothing looked at the
image, so the corpus holds 4,548 solid lines and 9 dashed ones, and the nine were drawn by hand. That is not
a weak classifier, it is the absence of one, and it disabled the distinction the event layer rests on:
crossing a solid line is an offence and crossing a dashed one is an ordinary manoeuvre, so with everything
typed solid every crossing derived as a violation and the severity axis carried no information.

The type is visible in the pixels. Sampling perpendicular to the curve at points along it gives a strip, and
the strip says which line this is:

  solid      paint present along nearly the whole run
  dashed     paint alternating with road at roughly regular intervals
  double     two paint ridges side by side in the lateral profile
  road_edge  no paint ridge at all, but the surface differs across the line

Run lengths rather than a Fourier or autocorrelation period, deliberately. A dashed lane is foreshortened by
perspective, so its period in image space shrinks toward the horizon and no single dominant frequency
exists; the alternation still does. Counting runs and asking whether they are regular survives the
foreshortening that a period estimate does not.

That regularity test is the load-bearing part and the reason this is not a duty-cycle threshold. A solid line
half hidden behind a car has a duty cycle around 0.5, exactly like a dashed one. What separates them is that
dashes are evenly spaced and occlusion is not, and getting this wrong invents a permission to cross.

Distinct from `marking.py` in this package, which pulls lane *geometry* out of a segmentation mask. This
takes geometry as given and asks what the line is.

Everything above `classify_lane` is pure over arrays, so every branch is reachable from a constructed
profile, including the ones whose whole job is to refuse: the occluded solid that must not read dashed, and
the worn line that must read unknown rather than guess.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from core.logging import get_logger

log = get_logger("lane_linetype")

MODEL_VERSION = "linetype-profile-v1"

# Perspective compresses dashes toward the horizon until they are shorter than a pixel and no rule can see
# them. Only the near part of the curve is analysed, measured from the bottom of its own extent.
NEAR_FRACTION = 0.6
# Samples along that near segment. Enough to resolve several dashes, few enough that a short lane still
# yields a usable profile.
N_ALONG = 64
# Half-width of the perpendicular strip, as a fraction of frame width. A lane fit is rarely more than a few
# pixels off the paint, and a wide strip starts including the neighbouring lane.
LATERAL_FRAC = 0.012
# How much of that strip counts as "the line" rather than "the road beside it". Wide enough to hold both
# ribbons of a double line, since a double whose second ribbon landed in the flank would raise the road level
# and hide itself.
CENTRE_FRAC = 0.6
# Contrast below this, in 0..255 after subtracting the local road level, is not paint. It is asphalt noise.
MIN_PAINT_CONTRAST = 12.0
# A run of paint or gap shorter than this fraction of the profile is speckle rather than a dash.
MIN_RUN_FRAC = 0.02
# Dashes must alternate at least this many times before regularity means anything.
MIN_PAINT_RUNS_FOR_DASHED = 3
# Coefficient of variation of the gap lengths. Evenly spaced dashes sit well under this; the gaps a car or a
# shadow cuts into a solid line do not.
MAX_GAP_CV_FOR_DASHED = 0.55
# Duty cycle above this, with no regular alternation, is a solid line.
SOLID_DUTY = 0.80
# Paint covering at least this much of the run, broken fewer times than dashes ever are, is a continuous line
# with something in front of it. A car sitting on a kerb line is the ordinary case and it must not fall
# through to unknown, or the commonest solid line in traffic becomes unclassifiable.
MIN_DUTY_FOR_OCCLUDED_SOLID = 0.40

VALID_TYPES = ("solid", "dashed", "double", "road_edge", "implicit", "fallback", "unknown")


@dataclass
class Evidence:
    """Why the classifier said what it said. Stored on the lane so a disputed type can be argued with."""

    contrast: float = 0.0
    duty: float = 0.0
    paint_runs: int = 0
    gap_cv: float | None = None
    lateral_peaks: int = 0
    peak_separation_px: float | None = None
    cross_surface_delta: float = 0.0
    samples: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"contrast": round(self.contrast, 2), "duty": round(self.duty, 3),
                "paint_runs": self.paint_runs,
                "gap_cv": None if self.gap_cv is None else round(self.gap_cv, 3),
                "lateral_peaks": self.lateral_peaks,
                "peak_separation_px": (None if self.peak_separation_px is None
                                       else round(self.peak_separation_px, 1)),
                "cross_surface_delta": round(self.cross_surface_delta, 2),
                "samples": self.samples, "notes": list(self.notes)}


def resample_curve(control_points: list, n: int = N_ALONG,
                   near_fraction: float = NEAR_FRACTION) -> np.ndarray:
    """Evenly spaced points along the near part of the lane, plus the local direction at each.

    Returns an (n, 4) array of x, y, dx, dy. Near is measured down the lane's own vertical extent: a lane
    runs from the bottom of the frame toward the horizon, and the bottom portion is where paint is
    resolvable at all. Arc-length spacing rather than spacing in y, so a steeply angled lane is not sampled
    more densely than an upright one.
    """
    pts = np.array([[float(p[0]), float(p[1])] for p in (control_points or [])
                    if p is not None and len(p) >= 2], dtype=np.float64)
    if len(pts) < 2:
        return np.empty((0, 4))
    pts = pts[np.argsort(pts[:, 1])]  # top of image first

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return np.empty((0, 4))

    # Densify the whole polyline first, then keep the near part of *that*. Cutting the control points
    # directly cannot work: a lane stored as two endpoints, which is most of them, has no control point in
    # the near half to keep, so the filter fell through and the curve was sampled all the way to the
    # horizon, where dashes are sub-pixel and the profile is noise.
    dense = np.linspace(0.0, cum[-1], max(4 * n, 32))
    dx_all = np.interp(dense, cum, pts[:, 0])
    dy_all = np.interp(dense, cum, pts[:, 1])

    y_hi = float(dy_all.max())
    span = y_hi - float(dy_all.min())
    if span <= 0:
        return np.empty((0, 4))
    near = dy_all >= (y_hi - span * near_fraction)
    if near.sum() < 2:
        near = np.ones_like(dy_all, dtype=bool)

    kx, ky = dx_all[near], dy_all[near]
    kseg = np.linalg.norm(np.diff(np.stack([kx, ky], axis=1), axis=0), axis=1)
    kcum = np.concatenate([[0.0], np.cumsum(kseg)])
    if kcum[-1] <= 0:
        return np.empty((0, 4))
    targets = np.linspace(0.0, kcum[-1], n)
    xs = np.interp(targets, kcum, kx)
    ys = np.interp(targets, kcum, ky)

    d = np.gradient(np.stack([xs, ys], axis=1), axis=0)
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return np.concatenate([np.stack([xs, ys], axis=1), d / norm], axis=1)


def sample_strip(gray: np.ndarray, curve: np.ndarray, half_width: int) -> np.ndarray:
    """The perpendicular strip around the curve: a row per point along it, a column per lateral offset.

    Out-of-frame samples come back NaN rather than clamped. Clamping repeats the edge pixel and manufactures
    a bright constant run at exactly the place a lane leaves the image, which then reads as paint.
    """
    if curve.size == 0 or half_width < 1:
        return np.empty((0, 0))
    h, w = gray.shape[:2]
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    # Perpendicular to the direction (dx, dy) is (-dy, dx).
    px = curve[:, 0][:, None] + offsets[None, :] * (-curve[:, 3])[:, None]
    py = curve[:, 1][:, None] + offsets[None, :] * (curve[:, 2])[:, None]

    xi = np.round(px).astype(np.int64)
    yi = np.round(py).astype(np.int64)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.full(xi.shape, np.nan, dtype=np.float64)
    out[inside] = gray[yi[inside], xi[inside]]
    return out


def paint_response(strip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How much a paint ridge rises above the road on *both* sides of it, per point along the line.

    A ridge, not a step. Taking the brightest pixel across the strip and calling the excess paint cannot tell
    a painted line from the edge of a lighter surface: both are bright on one side. So the centre band is
    compared against the road on each flank separately and the smaller of the two differences wins, which is
    positive only when the middle is brighter than what lies on either side. A kerb where asphalt meets
    concrete then scores near zero and falls through to the surface test, where it belongs.

    Flank level is a median rather than a mean so a car, a shadow or a second line clipping one flank moves
    it hardly at all.
    """
    if strip.size == 0:
        return np.empty(0), np.empty(0)
    cols = strip.shape[1]
    half = cols // 2
    band = max(1, int(round(half * CENTRE_FRAC)))
    lo, hi = max(0, half - band), min(cols, half + band + 1)
    if lo == 0 or hi == cols:
        # No room for flanks. Everything would be centre and the ridge test is meaningless.
        return np.zeros(strip.shape[0]), np.zeros(cols)

    with np.errstate(all="ignore"), warnings.catch_warnings():
        # A lane close to the frame edge has a flank entirely outside the image, so its median is over an
        # all-NaN slice. That is expected rather than exceptional, and fmax below already falls back to the
        # flank that is visible, so the warning is noise on a handled case.
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmax(strip[:, lo:hi], axis=1)
        left = np.nanmedian(strip[:, :lo], axis=1)
        right = np.nanmedian(strip[:, hi:], axis=1)
        # The higher flank is the harder test, and requiring the centre to clear it is what rejects a step.
        # fmax rather than maximum: with one flank off the image, the visible side is the whole test.
        along = centre - np.fmax(left, right)
        lateral = np.nanmean(strip - np.nanmedian(strip, axis=1, keepdims=True), axis=0)
    return np.nan_to_num(along, nan=0.0), np.nan_to_num(lateral, nan=0.0)


def runs_of(mask: np.ndarray) -> tuple[list[int], list[int]]:
    """Lengths of the True runs and of the False runs, each in order of appearance."""
    if mask.size == 0:
        return [], []
    changes = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate([[0], changes, [mask.size]])
    lengths = np.diff(bounds)
    values = mask[bounds[:-1]]
    return ([int(n) for n, v in zip(lengths, values, strict=True) if v],
            [int(n) for n, v in zip(lengths, values, strict=True) if not v])


def drop_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Flip runs too short to be a dash or a gap, so speckle does not count as alternation."""
    if mask.size == 0 or min_len <= 1:
        return mask
    out = mask.copy()
    changes = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate([[0], changes, [mask.size]])
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if (b - a) < min_len:
            out[a:b] = not mask[a]
    return out


def _groups(above: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return []
    out, start = [], int(idx[0])
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        if b != a + 1:
            out.append((start, int(a)))
            start = int(b)
    out.append((start, int(idx[-1])))
    return out


def lateral_peaks(lateral: np.ndarray, contrast: float) -> tuple[int, float | None]:
    """How many paint ridges sit side by side across the line, and how far apart they are.

    A double line is two ridges with a trough between them. Requiring that trough to fall to half the weaker
    ridge is what stops one wide, slightly lumpy line from reading as two.
    """
    if lateral.size < 5 or contrast < MIN_PAINT_CONTRAST:
        return 0, None
    thr = max(MIN_PAINT_CONTRAST, contrast * 0.45)
    groups = _groups(lateral >= thr)
    if len(groups) < 2:
        return len(groups), None

    peaks = [float(lateral[a:b + 1].max()) for a, b in groups]
    order = np.argsort(peaks)[::-1][:2]
    i, j = sorted((int(order[0]), int(order[1])))
    lo, hi = groups[i][1], groups[j][0]
    if hi <= lo:
        return len(groups), None
    if float(lateral[lo:hi + 1].min()) > min(peaks[i], peaks[j]) * 0.5:
        return 1, None  # one lumpy ridge, not two lines
    centres = [(a + b) / 2.0 for a, b in groups]
    return 2, abs(centres[j] - centres[i])


def interior_gaps(mask: np.ndarray, gap_runs: list[int]) -> list[int]:
    """Gaps between dashes, excluding the ones at the ends.

    A gap at either end is the lane running out of frame or out of annotation, not a gap between two dashes,
    and counting it makes evenly spaced dashes look irregular.
    """
    gaps = list(gap_runs)
    if mask.size and not mask[0] and gaps:
        gaps = gaps[1:]
    if mask.size and not mask[-1] and gaps:
        gaps = gaps[:-1]
    return gaps


def classify_profile(along: np.ndarray, lateral: np.ndarray, *,
                     cross_surface_delta: float = 0.0) -> tuple[str, float, Evidence]:
    """The decision, from the two profiles alone. Pure, so every branch is reachable in a test."""
    ev = Evidence(samples=int(along.size), cross_surface_delta=float(cross_surface_delta))
    if along.size < 8:
        ev.notes.append("too few samples along the curve to read the paint")
        return "unknown", 0.0, ev

    # How far the paint rises above the road, not how much it varies along the run. Using the spread was the
    # first version and it was exactly wrong: a solid line is uniformly bright, so its spread is zero and it
    # read as bare asphalt.
    contrast = float(np.percentile(along, 90))
    ev.contrast = contrast

    if contrast < MIN_PAINT_CONTRAST:
        # No paint. Either an unmarked boundary somebody drew, or a real road edge where the surface itself
        # changes. The difference across the line is what tells those apart.
        ev.notes.append("no paint ridge above the surrounding road")
        if cross_surface_delta >= MIN_PAINT_CONTRAST:
            return "road_edge", round(min(1.0, 0.4 + cross_surface_delta / 60.0), 3), ev
        ev.notes.append("and no surface change across it either")
        return "unknown", 0.15, ev

    peaks, sep = lateral_peaks(lateral, contrast)
    ev.lateral_peaks, ev.peak_separation_px = peaks, sep

    # Half way between the road (zero, by construction) and the paint.
    mask = drop_short_runs(along >= contrast * 0.5,
                           max(1, int(round(along.size * MIN_RUN_FRAC))))
    paint_runs, gap_runs = runs_of(mask)
    ev.duty, ev.paint_runs = float(mask.mean()), len(paint_runs)

    gaps = interior_gaps(mask, gap_runs)
    if len(gaps) >= 2:
        g = np.array(gaps, dtype=np.float64)
        ev.gap_cv = float(g.std() / g.mean()) if g.mean() > 0 else None

    if peaks >= 2:
        # Two ridges is a double line whatever the along-profile says. A broken double is still a double.
        return "double", round(min(1.0, 0.55 + contrast / 120.0), 3), ev

    regular = ev.gap_cv is not None and ev.gap_cv <= MAX_GAP_CV_FOR_DASHED
    if len(paint_runs) >= MIN_PAINT_RUNS_FOR_DASHED and ev.duty < SOLID_DUTY and regular:
        conf = 0.45 + 0.08 * len(paint_runs) + 0.25 * (1.0 - (ev.gap_cv or 0.0))
        return "dashed", round(min(1.0, conf), 3), ev

    if ev.duty >= SOLID_DUTY:
        return "solid", round(min(1.0, 0.55 + contrast / 120.0), 3), ev

    if len(paint_runs) >= MIN_PAINT_RUNS_FOR_DASHED and not regular:
        # Alternating but not evenly: a solid line with things in front of it, which is what a row of parked
        # cars does to the profile. Calling that dashed would invent a permission to cross.
        ev.notes.append("paint is interrupted but not at regular intervals, which is occlusion not dashes")
        return "solid", round(min(0.75, 0.35 + contrast / 160.0), 3), ev

    if 0 < len(paint_runs) < MIN_PAINT_RUNS_FOR_DASHED and ev.duty >= MIN_DUTY_FOR_OCCLUDED_SOLID:
        # Too few alternations to be dashes at all. Over the near segment a dashed lane shows at least three
        # of them, so one or two stretches of paint with gaps between is a continuous line partly hidden.
        # Confidence tracks how much of it was actually visible.
        ev.notes.append("too few breaks to be dashes, so a continuous line partly hidden")
        return "solid", round(min(0.7, 0.25 + 0.5 * ev.duty), 3), ev

    ev.notes.append("paint present but neither continuous nor regularly broken")
    return "unknown", 0.25, ev


def cross_surface_difference(strip: np.ndarray) -> float:
    """How different the surface is on one side of the line from the other.

    A road edge carries no paint but does separate two materials, so the halves of the strip differ. A lane
    drawn down the middle of open asphalt does not.
    """
    if strip.size == 0 or strip.shape[1] < 4:
        return 0.0
    half = strip.shape[1] // 2
    with np.errstate(all="ignore"):
        left, right = np.nanmedian(strip[:, :half]), np.nanmedian(strip[:, half + 1:])
    if np.isnan(left) or np.isnan(right):
        return 0.0
    return float(abs(left - right))


def classify_lane(image_bgr, control_points: list, *,
                  frame_width: int | None = None) -> tuple[str, float, Evidence]:
    """Read one lane's type off the frame it was drawn on."""
    import cv2

    if image_bgr is None:
        return "unknown", 0.0, Evidence(notes=["the frame image could not be read"])
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    half = max(2, int(round(LATERAL_FRAC * int(frame_width or gray.shape[1]))))

    curve = resample_curve(control_points)
    if curve.size == 0:
        return "unknown", 0.0, Evidence(notes=["the lane has too few control points to sample"])

    strip = sample_strip(gray, curve, half)
    along, lateral = paint_response(strip)
    return classify_profile(along, lateral, cross_surface_delta=cross_surface_difference(strip))
