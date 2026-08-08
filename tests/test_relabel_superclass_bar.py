"""Crossing a superclass is a different claim from refining within one, and needs a different bar.

Measured against the 302 human-accepted objects in this corpus, which are the best ground truth available:
a label a person verified and the agent wants to change is very likely the agent being wrong. The original
single-threshold rule would have changed 73 of them (24.2%), 50 across a superclass, including
`motorcycle -> pedestrian` at 0.744 and `motorcycle -> street_vendor` at 0.747, both auto-kept without any
human seeing them.

The corpus already said this would happen. A judge calibrated against human review rulings put label
precision at 0.958 by superclass with only 2 of 285 errors crossing one, so the superclass is the part of a
label that is almost always right, and a model proposing to cross it is usually the thing that is wrong.

With the two bars: 25 changes (8.3%), 10 auto-kept, and cross-superclass down from 50 to 2. The 23
within-superclass refinements are all still proposed, which is the point: the rule is meant to remove a
failure mode, not to make the agent timid.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.agent.relabel_agent import (
    CROSS_L1_CONF,
    CROSS_L1_MARGIN,
    MIN_CROP_PX,
    _decide,
)

# Ontology ids used below, resolved once so the test reads as a statement about classes rather than numbers.
from services.autolabel.ontology import get_ontology

ONTO = get_ontology()
MOTORCYCLE = ONTO.by_name("motorcycle").id      # l1 two_wheeler
PEDESTRIAN = ONTO.by_name("pedestrian").id      # l1 vru
SEDAN = ONTO.by_name("sedan").id                # l1 four_wheeler
SUV = ONTO.by_name("suv").id                    # l1 four_wheeler

CROP = np.full((64, 64, 3), 128, np.uint8)


def _preds(monkeypatch, ranked: list[tuple[int, float]]):
    """Pin the classifier's output so the decision rule can be tested without a GPU or an image."""
    out = [{"class_id": cid, "class_name": ONTO.by_id(cid).name, "conf": c} for cid, c in ranked]
    monkeypatch.setattr("services.autolabel.classify_crop.classify_crop", lambda *a, **k: out)


DEFAULTS = {"min_conf": 0.45, "margin": 0.15, "strong_conf": 0.60, "strong_margin": 0.30}
# Auto-keep is opt-in and off by default, so the tests that are about the keep-versus-review split have to
# ask for it. See test_nothing_is_auto_kept_by_default for why the default is what it is.
KEEPING = {**DEFAULTS, "auto_keep": True}


def test_a_confident_within_superclass_refinement_is_still_kept(monkeypatch):
    """The rule must not make the agent timid: sedan to suv is the distinction the taxonomy exists for."""
    _preds(monkeypatch, [(SUV, 0.80), (SEDAN, 0.10)])
    out = _decide(CROP, SEDAN, **KEEPING)
    assert out is not None
    assert out[0] == SUV and out[3] == "relabel_keep"


def test_a_cross_superclass_change_at_the_old_threshold_is_now_refused(monkeypatch):
    """The exact shape that produced motorcycle -> street_vendor at 0.747, auto-kept."""
    _preds(monkeypatch, [(PEDESTRIAN, 0.747), (MOTORCYCLE, 0.10)])
    assert _decide(CROP, MOTORCYCLE, **DEFAULTS) is None


def test_a_cross_superclass_change_needs_the_higher_bar(monkeypatch):
    _preds(monkeypatch, [(PEDESTRIAN, CROSS_L1_CONF - 0.01), (MOTORCYCLE, 0.05)])
    assert _decide(CROP, MOTORCYCLE, **DEFAULTS) is None


def test_a_cross_superclass_change_that_clears_the_bar_always_goes_to_review(monkeypatch):
    """It may be right. It is not right often enough to change the corpus without somebody looking."""
    _preds(monkeypatch, [(PEDESTRIAN, 0.96), (MOTORCYCLE, 0.02)])
    out = _decide(CROP, MOTORCYCLE, **DEFAULTS)
    assert out is not None
    assert out[3] == "relabel_review", "a cross-superclass change is never auto-kept"


def test_the_cross_superclass_margin_is_enforced_too(monkeypatch):
    """High confidence on both means the model is not actually discriminating between them."""
    _preds(monkeypatch, [(PEDESTRIAN, 0.95), (MOTORCYCLE, 0.90)])
    assert _decide(CROP, MOTORCYCLE, **DEFAULTS) is None


def test_the_cross_bar_is_meaningfully_stricter_than_the_within_bar():
    """If they converged the rule would be decoration."""
    assert CROSS_L1_CONF > DEFAULTS["strong_conf"]
    assert CROSS_L1_MARGIN > DEFAULTS["strong_margin"]


def test_a_tiny_crop_is_declined_whatever_the_model_says(monkeypatch):
    """A distant object is exactly where a zero-shot confidence is least earned."""
    _preds(monkeypatch, [(SUV, 0.99), (SEDAN, 0.01)])
    tiny = np.full((MIN_CROP_PX - 1, MIN_CROP_PX - 1, 3), 128, np.uint8)
    assert _decide(tiny, SEDAN, **DEFAULTS) is None


def test_agreement_with_the_current_label_proposes_nothing(monkeypatch):
    _preds(monkeypatch, [(SEDAN, 0.9), (SUV, 0.05)])
    assert _decide(CROP, SEDAN, **DEFAULTS) is None


def test_a_fallback_bucket_is_never_proposed(monkeypatch):
    """Relabelling a specific class down into a catch-all loses information rather than adding any."""
    fb = ONTO.by_name("vehicle_fallback").id
    _preds(monkeypatch, [(fb, 0.99), (SEDAN, 0.01)])
    assert _decide(CROP, SEDAN, **DEFAULTS) is None


@pytest.mark.parametrize("conf,gap_from,expected", [
    (0.80, 0.10, "relabel_keep"),     # decisive and clear
    (0.50, 0.30, "relabel_review"),   # moderate: a human confirms
])
def test_within_superclass_still_splits_keep_from_review(monkeypatch, conf, gap_from, expected):
    _preds(monkeypatch, [(SUV, conf), (SEDAN, gap_from)])
    out = _decide(CROP, SEDAN, **KEEPING)
    assert out is not None and out[3] == expected


def test_nothing_is_auto_kept_by_default(monkeypatch):
    """The measurement that set the default. Against the 302 objects a person verified, all 10 changes this
    would have applied unreviewed overruled that person, among them traffic_sign -> milestone at 0.985. The
    confidence is a softmax over how well a crop matches a class name, so it is highest exactly where two
    names are close, which is where the agent is least trustworthy."""
    _preds(monkeypatch, [(SUV, 0.99), (SEDAN, 0.001)])
    out = _decide(CROP, SEDAN, **DEFAULTS)
    assert out is not None, "it should still propose the change"
    assert out[3] == "relabel_review", "but a person decides it"
