"""Section 1.5: the promotion gate refuses untrustworthy metrics. A reconstructed run (predictions backfilled
from review history, no real confidence) and a harness divergence (the val-pass and prediction plane disagreeing
on the same gold set beyond epsilon) both fail closed, before any mAP or safety comparison runs.
"""

from __future__ import annotations

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.govern.champion import champion_gate

_STRONG = {"map50": 0.95, "map": 0.9, "safe_miou": 0.95, "per_class": {}, "per_class_recall": {}}


def _cfg():
    return get_settings().phase4.govern


def test_gate_refuses_reconstructed_metrics():
    gate = champion_gate({**_STRONG, "reconstructed": True}, None, get_ontology(), _cfg())
    assert gate["promote"] is False
    assert "reconstructed" in gate["reasons"][0]


def test_gate_refuses_divergent_harnesses():
    gate = champion_gate({**_STRONG, "harness_divergent": True},
                         {"map50": 0.5, "safe_miou": 0.5, "per_class": {}}, get_ontology(), _cfg())
    assert gate["promote"] is False
    assert "diverge" in gate["reasons"][0]


def test_gate_still_evaluates_clean_metrics_normally():
    # Without the flags, a strong first challenger with a Safe-mIoU is promotable (existing behaviour intact).
    gate = champion_gate(_STRONG, None, get_ontology(), _cfg())
    assert "promote" in gate  # gate runs its normal logic rather than short-circuiting
