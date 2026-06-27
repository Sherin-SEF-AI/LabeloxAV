"""Confidence calibration. Turns raw path scores plus the agreement signal into a single
trustworthy probability so the gate thresholds mean something (Principle 04).

Temperature scaling to start; isotonic regression once a human-verified gold set exists (M1+ of
the loop, deferred). The agreement bonus and disagreement penalty encode the core idea: consensus
across paths is more trustworthy than any single score.
"""

from __future__ import annotations

import math

from core.config import CalibrateSettings


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def calibrate_confidence(
    raw_conf: float,
    agreement: bool,
    class_disagreement: bool,
    mask_box_disagree: bool,
    cfg: CalibrateSettings,
) -> float:
    """Return a calibrated confidence in [0, 1].

    Temperature scaling sharpens or softens the raw score; a temperature > 1 is conservative
    (pulls scores toward 0.5), which is the safe default before a gold set is available. Once a gold
    set has been used to fit an isotonic curve (method=isotonic + isotonic_uri), the calibrated base
    becomes the empirical P(correct) for that raw score, so the gate's 0.95 means real precision.
    """
    if cfg.method == "isotonic" and cfg.isotonic_uri:
        try:
            from services.autolabel.isotonic import apply_isotonic

            cal = apply_isotonic(cfg.isotonic_uri, raw_conf)
        except Exception:  # noqa: BLE001 - fall back to temperature if the curve cannot be loaded
            cal = _sigmoid(_logit(raw_conf) / max(cfg.temperature, 1e-6))
    else:
        cal = _sigmoid(_logit(raw_conf) / max(cfg.temperature, 1e-6))

    if agreement:
        cal += cfg.agreement_bonus
    if class_disagreement:
        cal -= cfg.disagreement_penalty
    if mask_box_disagree:
        cal -= cfg.disagreement_penalty * 0.5

    return float(min(max(cal, 0.0), 1.0))
