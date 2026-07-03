"""M-F.1 label quality score: the factor computation and the confidence-anchored combination with hard-defect
penalties and the human-verdict override. The VLM-judged gold correlation is validated operationally (see
services/analytics/quality_score.validate); this pins the deterministic scoring logic."""

from __future__ import annotations

from services.analytics.quality_score import compute_factors, score_from_factors


def test_confidence_source_of_the_factor():
    # a model object with a calibrated_from scalar -> the confidence factor is the calibrated value
    f = compute_factors({"calibrated_from": 0.9, "agreement": True}, 0.9, None, None, "fused")
    assert 0.0 <= f["confidence"] <= 1.0
    # a human object has no model score -> fixed human-authority confidence, not its stored conf
    fh = compute_factors({}, 1.0, None, None, "human")
    assert fh["confidence"] == 0.9


def test_penalties_and_flags():
    base = compute_factors({"calibrated_from": 0.9, "agreement": True}, None, None, None, "fused")
    assert base["agreement"] == 1.0 and base["geometry"] == 1.0 and base["rig"] == 1.0
    flagged = compute_factors({"calibrated_from": 0.9, "agreement": False, "mask_box_disagree": True,
                               "quality_flags": ["impossible_size"]}, None, None, True, "fused")
    assert flagged["agreement"] == 0.45 and flagged["geometry"] == 0.15 and flagged["mask"] == 0.35 and flagged["rig"] == 0.2


def test_score_penalises_defects_and_honours_verdict():
    clean = {"confidence": 0.48, "agreement": 1.0, "geometry": 1.0, "mask": 1.0, "temporal": 1.0, "rig": 1.0}
    geom_bad = {**clean, "geometry": 0.15}
    rig_bad = {**clean, "rig": 0.2}
    s_clean = score_from_factors(clean, "auto_accept", "fused")
    s_geom = score_from_factors(geom_bad, "auto_accept", "fused")
    s_rig = score_from_factors(rig_bad, "auto_accept", "fused")
    assert s_clean == 0.48                       # confidence-anchored, no penalty
    assert s_geom < s_clean and s_rig < s_clean  # a geometric or cross-view defect lowers quality

    # human verdict dominates: a rejected label is low quality, an accepted human one is high
    assert score_from_factors(clean, "rejected", "fused") <= 0.20
    assert score_from_factors(clean, "accepted", "human") >= 0.75


def test_score_ranks_by_confidence():
    # with everything else equal, a higher calibrated confidence yields a higher quality score
    lo = score_from_factors({"confidence": 0.2, "agreement": 1.0, "geometry": 1.0, "mask": 1.0, "temporal": 1.0, "rig": 1.0}, None, None)
    hi = score_from_factors({"confidence": 0.45, "agreement": 1.0, "geometry": 1.0, "mask": 1.0, "temporal": 1.0, "rig": 1.0}, None, None)
    assert hi > lo
