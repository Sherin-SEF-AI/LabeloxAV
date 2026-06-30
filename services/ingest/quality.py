"""Cheap per-frame quality gate. Runs on CPU before any GPU touches the data, which is the
highest-leverage cost saving in the system: rejecting junk here avoids labeling it.

Blur via variance-of-Laplacian, exposure via mean luma and clipped-pixel fraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.config import IngestSettings


@dataclass
class QualityResult:
    blur: float          # variance of Laplacian; higher is sharper
    mean_luma: float     # 0..255
    clip_fraction: float # fraction of pixels at 0 or 255
    score: float         # 0..1 normalized overall quality
    accepted: bool
    reasons: list[str]


def score_frame(image_bgr: np.ndarray, cfg: IngestSettings) -> QualityResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_luma = float(gray.mean())
    total = gray.size
    clipped = int(np.count_nonzero(gray <= 2) + np.count_nonzero(gray >= 253))
    clip_fraction = clipped / max(total, 1)

    reasons: list[str] = []
    # Noise/corruption: extreme high-frequency energy is a garbage frame (random static), not a sharp image.
    # Checked first so a corrupted frame is rejected rather than scored as maximally sharp.
    is_noise = blur > cfg.noise_blur_threshold
    if is_noise:
        reasons.append(f"noise/corrupted blur {blur:.0f} > {cfg.noise_blur_threshold:.0f}")
    if blur < cfg.blur_threshold:
        reasons.append(f"blur {blur:.1f} < {cfg.blur_threshold}")
    if mean_luma < cfg.exposure_low:
        reasons.append(f"underexposed luma {mean_luma:.1f}")
    if mean_luma > cfg.exposure_high:
        reasons.append(f"overexposed luma {mean_luma:.1f}")
    if clip_fraction > cfg.clip_fraction_max:
        reasons.append(f"clipping {clip_fraction:.2f} > {cfg.clip_fraction_max}")

    # Normalized score: sharpness saturating near a healthy blur value, penalized by clipping
    # and by distance of luma from the mid-range.
    sharp_term = 0.0 if is_noise else min(1.0, blur / (cfg.blur_threshold * 4.0))
    luma_mid = (cfg.exposure_low + cfg.exposure_high) / 2.0
    luma_span = (cfg.exposure_high - cfg.exposure_low) / 2.0
    luma_term = max(0.0, 1.0 - abs(mean_luma - luma_mid) / max(luma_span, 1.0))
    clip_term = max(0.0, 1.0 - clip_fraction / max(cfg.clip_fraction_max, 1e-6))
    score = float(0.5 * sharp_term + 0.3 * luma_term + 0.2 * clip_term)

    return QualityResult(
        blur=blur,
        mean_luma=mean_luma,
        clip_fraction=clip_fraction,
        score=round(score, 4),
        accepted=len(reasons) == 0,
        reasons=reasons,
    )
