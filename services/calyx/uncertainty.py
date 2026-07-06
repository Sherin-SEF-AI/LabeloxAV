"""CALYX calibration uncertainty (M11). Every calibration carries a confidence in [0,1] derived from the
reprojection residual and the number of correspondences the estimate rested on. It propagates to a per-sample
confidence that ORACLYX (fusion) and the workspace read to weight how much to trust the 3D geometry: a sample
from a weakly-calibrated session contributes less to a fused pseudo-label. Pure and monotonic, so it is
testable."""

from __future__ import annotations

import math

_RESIDUAL_SCALE_PX = 3.0    # residual at which confidence is ~1/e
_N_HALF = 12                # correspondences at which the sample-count term is 0.5


def calibration_confidence(residual_px: float | None, n_correspondences: int) -> float:
    """Confidence in a calibration: high when the residual is small and it rested on many correspondences,
    low when the residual is large or the estimate is thin. Monotonic in both arguments."""
    if residual_px is None or n_correspondences <= 0:
        return 0.0
    resid_term = math.exp(-max(0.0, float(residual_px)) / _RESIDUAL_SCALE_PX)
    count_term = n_correspondences / (n_correspondences + _N_HALF)
    return round(float(resid_term * count_term), 4)


def sample_confidence(session_confidence: float, frame_quality: float | None = None) -> float:
    """Per-sample calibration confidence: the session calibration confidence, optionally attenuated by the
    frame's own quality (a blurry frame degrades the geometry further)."""
    q = 1.0 if frame_quality is None else max(0.0, min(1.0, float(frame_quality)))
    return round(float(session_confidence) * (0.5 + 0.5 * q), 4)
