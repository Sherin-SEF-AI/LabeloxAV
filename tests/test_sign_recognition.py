"""The sign classifier has to be able to say "not a sign".

Measured before this was written, feeding real crops from the corpus through `classify_sign`:

    fed a crop of        n  mean conf    min    max   most common type assigned
    traffic_sign        12      0.759  0.210  0.999   speed_limit
    bus                 12      0.817  0.513  0.990   bus_stop      <- higher than real signs
    truck               12      0.746  0.413  0.976   toll_ahead
    pedestrian          12      0.483  0.204  0.745   pedestrian_crossing
    random noise        12      0.167  0.152  0.185   no_horn

A photograph of a bus scored higher as a sign than an actual sign did. The cause is the prompts: SigLIP
matches "a bus stop information sign" to a bus, "a toll plaza ahead information board" to a truck, and
"a pedestrian crossing warning sign" to a pedestrian, because the object noun dominates the phrase. The
classifier was never distinguishing signs from the things signs depict.

Softmax over 21 sign prompts cannot express "none of these". Whatever crop arrives, the probabilities sum
to one and something wins. The fix is to give it somewhere to say no: negative prompts describing what a
sign is not, and a decision made on the margin between the best sign and the best negative rather than on a
probability that was never a measurement.

These tests use synthetic imagery rather than the corpus so they run without a GPU-loaded model where one is
not available, and skip cleanly when SigLIP cannot be loaded at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.autolabel.signs.taxonomy import get_sign_taxonomy


def _siglip_or_skip():
    try:
        from services.intelligence.embed import siglip2

        siglip2.encode_texts(["a traffic sign"])
        return siglip2
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"SigLIP 2 unavailable: {exc}")


def test_taxonomy_declares_negatives():
    """The taxonomy has to carry what a sign is not, or nothing downstream can reject."""
    tax = get_sign_taxonomy()
    negs = tax.get("negatives") or []
    assert len(negs) >= 8, "too few negative prompts to cover the things that get mistaken for signs"
    names = {n["name"] for n in negs}
    # Asserted by name rather than by words in the prompt: the prompt wording is the thing most likely to be
    # tuned later, and a test that breaks when "a pedestrian" becomes "a person walking" is testing phrasing
    # rather than coverage. The four below are the ones the measurement caught mistyped, plus the back of a
    # sign, which Mapillary imports under the same class as the front.
    for needed in ("neg_bus", "neg_truck", "neg_pedestrian", "neg_sign_back", "neg_hoarding"):
        assert needed in names, f"no negative covers {needed}, which was seen mistyped as a sign"
    assert all(n["prompt"].strip() for n in negs), "a negative with an empty prompt rejects nothing"


def test_negative_prompts_are_not_also_types():
    """A negative that is also a sign type would reject the very thing it describes."""
    tax = get_sign_taxonomy()
    names = {t["name"] for t in tax["types"]}
    for n in tax.get("negatives") or []:
        assert n["name"] not in names, f"{n['name']} is both a sign type and a negative"


def test_classify_returns_none_for_a_flat_untextured_crop():
    """A blank grey patch is not a sign, and saying so is the whole point of this change."""
    _siglip_or_skip()
    from services.autolabel.signs.recognize import classify_sign

    grey = np.full((80, 80, 3), 128, dtype=np.uint8)
    res = classify_sign(grey)
    assert res["sign_type"] is None
    assert res["sign_category"] is None
    assert res["rejected"] is True
    assert res["reason"]


def test_classify_returns_none_for_noise():
    """Random pixels previously came back as `no_horn`. They must now come back as nothing."""
    _siglip_or_skip()
    from services.autolabel.signs.recognize import classify_sign

    rng = np.random.default_rng(0)
    for _ in range(3):
        res = classify_sign(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        assert res["sign_type"] is None, f"noise was typed as {res['sign_type']}"


def test_a_confident_result_carries_its_margin_not_a_softmax():
    """Whatever survives has to report the evidence it survived on.

    The old `conf` was a softmax over 21 mutually exclusive prompts, which is a number that exists for any
    input at all. The margin against the best negative is the quantity the decision is actually made on, so
    it is the one that gets recorded.
    """
    _siglip_or_skip()
    from services.autolabel.signs.recognize import classify_sign

    res = classify_sign(np.full((80, 80, 3), 128, dtype=np.uint8))
    assert "margin" in res
    assert isinstance(res["margin"], float)


def test_octagonal_red_sign_beats_a_flat_patch():
    """A crude drawn STOP sign should sit further from the negatives than a blank patch does.

    Deliberately a relative assertion. Pinning an absolute score to a synthetic drawing would encode this
    machine's model build rather than the behaviour, and would break on any model change for no good reason.
    """
    _siglip_or_skip()
    import cv2

    from services.autolabel.signs.recognize import sign_margin

    sign = np.full((96, 96, 3), 255, dtype=np.uint8)
    pts = np.array([[(48 + 40 * np.cos(a), 48 + 40 * np.sin(a))
                     for a in np.linspace(0, 2 * np.pi, 9)[:-1]]], dtype=np.int32)
    cv2.fillPoly(sign, pts, (30, 30, 200))
    cv2.putText(sign, "STOP", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    blank = np.full((96, 96, 3), 128, dtype=np.uint8)

    assert sign_margin(sign) > sign_margin(blank)


def test_text_bearing_only_for_types_that_carry_text():
    """OCR routing keys off this flag, so it has to mean what it says."""
    tax = get_sign_taxonomy()
    by_name = {t["name"]: t for t in tax["types"]}
    assert by_name["speed_limit"]["text_bearing"] is True
    assert by_name["destination_board"]["text_bearing"] is True
    assert by_name["stop"]["text_bearing"] is False
    assert by_name["no_horn"]["text_bearing"] is False
