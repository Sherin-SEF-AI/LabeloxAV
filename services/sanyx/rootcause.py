"""SANYX root-cause classification (M10). A HealthReport says a session is degraded or quarantined; this
names WHY, from the same check evidence, and hands the operator a remediation hint. Each fault has a
characteristic evidence signature (a loose GMSL2 connector shows as camera sequence gaps and PTS skew; a
fouled lens shows as low high-frequency energy and persistent occlusion; sun glare as transient overexposure;
urban-canyon multipath as high HDOP with RTK loss; IMU thermal drift as a climbing bias with clock slope).

The shipped classifier is a deterministic signature matcher over the evidence, so it is testable on synthetic
known-fault signatures with no labeled fault corpus. train() is the hook that fits a learned model over the
same evidence features WHEN a labeled fault dataset exists; it is capability-gated (needs scikit-learn and
labels) and never fabricates a model."""

from __future__ import annotations

REMEDIATION = {
    "loose_gmsl2_connector": "reseat the GMSL2 coax at the camera and the deserializer; check the locking clip",
    "lens_fouling": "clean the lens; check for water, dust, or an obstruction in the persistent region",
    "sun_glare": "expected transient in low-sun conditions; consider a hood or route timing, not a hardware fault",
    "gps_urban_canyon_multipath": "GNSS degraded by buildings; rely on IMU dead-reckoning here, no rig action",
    "imu_thermal_drift": "let the MTi-3 thermally settle before capture; check enclosure ventilation",
    "time_sync_loss": "check the STM32 timesync pulse and wiring; the shared clock reference was lost",
}


def _ev(checks: list[dict]) -> dict:
    """Flatten the per-check evidence into one namespaced dict for signature matching."""
    out: dict = {}
    for c in checks:
        name = c.get("name")
        out[f"{name}.status"] = c.get("status")
        out[f"{name}.score"] = c.get("score")
        for k, v in (c.get("evidence") or {}).items():
            out[f"{name}.{k}"] = v
    return out


def _num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def classify(checks: list[dict]) -> dict:
    """Return the most likely fault {fault, confidence, remediation, signals} for a health report, or fault
    None when nothing is clearly wrong. Deterministic signature scoring over the evidence."""
    ev = _ev(checks)
    scores: dict[str, float] = {}
    signals: dict[str, list] = {}

    # time-sync pulse loss is unambiguous and dominates
    if ev.get("time_sync.pps_present") is False:
        return {"fault": "time_sync_loss", "confidence": 1.0,
                "remediation": REMEDIATION["time_sync_loss"], "signals": ["pps_present=false"]}

    # loose GMSL2 connector: dropped frames on a camera (sequence gaps) is the primary tell
    if ev.get("dropped_frames.status") in ("degraded", "quarantine"):
        wg = _num(ev.get("dropped_frames.worst_gap"))
        wr = _num(ev.get("dropped_frames.worst_drop_rate"))
        scores["loose_gmsl2_connector"] = min(1.0, 0.5 + wr * 5 + (wg >= 20) * 0.4)
        signals["loose_gmsl2_connector"] = [f"worst_gap={wg}", f"drop_rate={wr:.3f}"]

    # lens fouling: low high-frequency energy AND/OR persistent occlusion
    if ev.get("lens.status") in ("degraded", "quarantine"):
        occ = _num(ev.get("lens.occlusion_area_frac"))
        hf = _num(ev.get("lens.highfreq_ratio"), 1.0)
        scores["lens_fouling"] = min(1.0, 0.5 + occ * 2 + (hf < 0.002) * 0.3)
        signals["lens_fouling"] = [f"occlusion={occ:.2f}", f"highfreq={hf:.4f}"]

    # sun glare: exposure degraded, driven by clipping / low well-exposed fraction (transient, not a fault)
    if ev.get("exposure.status") in ("degraded", "quarantine"):
        we = _num(ev.get("exposure.well_exposed_frac"), 1.0)
        scores["sun_glare"] = min(1.0, 0.4 + (1.0 - we) * 0.8)
        signals["sun_glare"] = [f"well_exposed={we:.2f}"]

    # urban-canyon multipath: GPS degraded by HDOP + dropout, not a rig fault
    if ev.get("gps.status") in ("degraded", "quarantine"):
        hdop = _num(ev.get("gps.median_hdop"))
        drop = _num(ev.get("gps.longest_dropout_s"))
        scores["gps_urban_canyon_multipath"] = min(1.0, 0.4 + (hdop > 5) * 0.4 + (drop > 5) * 0.3)
        signals["gps_urban_canyon_multipath"] = [f"hdop={hdop}", f"dropout_s={drop}"]

    # IMU thermal drift: bias jump / saturation climbing, or a clock-slope tell
    if ev.get("imu.status") in ("degraded", "quarantine"):
        bias = _num(ev.get("imu.bias_jump"))
        sat = _num(ev.get("imu.saturation_frac"))
        scores["imu_thermal_drift"] = min(1.0, 0.4 + min(0.4, bias / 5) + sat * 3)
        signals["imu_thermal_drift"] = [f"bias_jump={bias:.2f}", f"saturation={sat:.3f}"]

    if not scores:
        return {"fault": None, "confidence": 0.0, "remediation": None, "signals": []}
    fault = max(scores, key=scores.get)
    return {"fault": fault, "confidence": round(scores[fault], 3),
            "remediation": REMEDIATION.get(fault), "signals": signals.get(fault, [])}


def feature_vector(checks: list[dict]) -> list[float]:
    """The numeric evidence features, for the learned classifier's training and inference."""
    ev = _ev(checks)
    keys = ["dropped_frames.worst_drop_rate", "dropped_frames.worst_gap", "lens.occlusion_area_frac",
            "lens.highfreq_ratio", "exposure.well_exposed_frac", "gps.median_hdop", "gps.longest_dropout_s",
            "imu.bias_jump", "imu.saturation_frac", "time_sync.clock_slope_ppm"]
    return [_num(ev.get(k)) for k in keys]


def train(labeled: list[tuple[list[dict], str]]):
    """Fit a learned root-cause model over the evidence features. Capability-gated: raises if scikit-learn is
    absent, and refuses on too few labels, rather than fabricating a model. Returns the fitted estimator."""
    import importlib.util
    if importlib.util.find_spec("sklearn") is None:
        raise RuntimeError("learned root-cause needs scikit-learn; the signature classifier is the default")
    if len(labeled) < 30:
        raise RuntimeError(f"need >= 30 labeled fault sessions to train, got {len(labeled)}")
    from sklearn.ensemble import RandomForestClassifier

    X = [feature_vector(ch) for ch, _ in labeled]
    y = [lbl for _, lbl in labeled]
    clf = RandomForestClassifier(n_estimators=200, random_state=7).fit(X, y)
    return clf
