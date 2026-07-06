"""CALYX cross-session calibration consensus (M11). Any single session's calibration is noisy; fusing the
calibrations from many sessions of the same rig gives a stronger prior than any one session, weighted by each
session's confidence. The fused prior tightens as sessions accumulate and is what a new session is validated
against. Pure and confidence-weighted, so it is testable against a known truth."""

from __future__ import annotations

import numpy as np


def fuse_calibrations(calibs: list[dict]) -> dict:
    """Confidence-weighted fusion of per-session calibrations into a rig prior. Each calib is
    {"rpy_deg": [3], "xyz_m": [3], "confidence": float}. Returns the fused extrinsic, an effective sample size,
    and the residual spread so a wide disagreement is visible rather than hidden."""
    valid = [c for c in calibs if c.get("confidence", 0) > 0 and c.get("rpy_deg") and c.get("xyz_m")]
    if not valid:
        return {"n": 0, "rpy_deg": None, "xyz_m": None, "confidence": 0.0}
    w = np.array([float(c["confidence"]) for c in valid])
    rpy = np.array([c["rpy_deg"] for c in valid], dtype=np.float64)
    xyz = np.array([c["xyz_m"] for c in valid], dtype=np.float64)
    wsum = w.sum()
    rpy_mean = (w[:, None] * rpy).sum(0) / wsum
    xyz_mean = (w[:, None] * xyz).sum(0) / wsum
    # effective sample size (Kish) and the confidence-weighted spread
    n_eff = float(wsum ** 2 / (w ** 2).sum())
    rpy_spread = float(np.sqrt((w[:, None] * (rpy - rpy_mean) ** 2).sum() / wsum))
    xyz_spread = float(np.sqrt((w[:, None] * (xyz - xyz_mean) ** 2).sum() / wsum))
    # prior confidence grows with effective sample size and shrinks with disagreement
    conf = float(min(1.0, n_eff / (n_eff + 3)) * np.exp(-rpy_spread / 2.0))
    return {"n": len(valid), "n_eff": round(n_eff, 2),
            "rpy_deg": [round(v, 4) for v in rpy_mean], "xyz_m": [round(v, 4) for v in xyz_mean],
            "rpy_spread_deg": round(rpy_spread, 4), "xyz_spread_m": round(xyz_spread, 4),
            "confidence": round(conf, 4)}
