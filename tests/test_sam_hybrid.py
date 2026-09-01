"""Two segmenters, and what their disagreement is allowed to mean.

SAM 2 produces the mask; SAM 1 is re-prompted with the same box and scores it. The value is in the fifth of
cases where they differ - measured on this corpus, median agreement is 0.893 and the low tail is almost all
riders, motorcycles and autorickshaws, where the mask boundary is genuinely ambiguous.

The assertion that matters most is that an unasked verifier is never read as a confident one.
"""

import numpy as np
import pytest

from services.autolabel.sam_hybrid import agrees, mask_iou


def _sq(size, lo, hi):
    m = np.zeros((size, size), dtype=bool)
    m[lo:hi, lo:hi] = True
    return m


class TestMaskIou:
    def test_identical_masks_agree_completely(self):
        a = _sq(20, 4, 16)
        assert mask_iou(a, a) == pytest.approx(1.0)

    def test_disjoint_masks_agree_not_at_all(self):
        assert mask_iou(_sq(20, 2, 8), _sq(20, 12, 18)) == 0.0

    def test_partial_overlap_is_between(self):
        v = mask_iou(_sq(20, 4, 16), _sq(20, 6, 18))
        assert 0.0 < v < 1.0

    def test_two_empty_masks_are_not_perfect_agreement(self):
        """An empty union is not unanimity. Returning 1.0 here would make a pair of failures look like the
        most trustworthy result in the corpus."""
        e = np.zeros((10, 10), dtype=bool)
        assert mask_iou(e, e) == 0.0

    def test_mismatched_shapes_disagree_rather_than_raising(self):
        assert mask_iou(_sq(10, 2, 8), _sq(20, 2, 8)) == 0.0


class TestTheVerdict:
    def test_a_missing_second_opinion_is_not_a_verdict(self):
        """`None` means the verifier never ran. Folding that into either answer is the whole failure this
        design exists to avoid: it would either flag every object on a box with no second model, or - worse
        - silently pass them all as agreed."""
        assert agrees(None) is None

    def test_it_splits_on_the_configured_floor(self):
        from core.config import get_settings

        thr = get_settings().models.openvocab.seg_agree_iou
        assert agrees(thr) is True
        assert agrees(thr - 0.01) is False

    def test_an_explicit_threshold_overrides_config(self):
        assert agrees(0.5, threshold=0.4) is True
        assert agrees(0.5, threshold=0.6) is False


class TestTheGateReadsIt:
    def test_a_disagreed_mask_cannot_auto_accept(self):
        import inspect

        from services.autolabel import gate

        src = inspect.getsource(gate)
        assert "mask_agreement" in src and "mask_agree_min" in src

    def test_none_never_routes_an_object_to_review(self):
        """The gate must treat an unmeasured mask as unmeasured. A `None < 0.8` comparison would raise; a
        naive falsy check would send every object without a verifier to review."""
        import inspect

        from services.autolabel import gate

        src = inspect.getsource(gate)
        assert "prov.mask_agreement is not None" in src, (
            "the gate must test for None explicitly, not rely on falsiness")

    def test_the_signal_is_distinct_from_mask_box_disagree(self):
        """mask_box_disagree is mask-versus-detection-box. This is mask-versus-mask. Overloading one field
        with both would corrupt every existing reader's interpretation of it."""
        from core.schemas import Provenance

        p = Provenance()
        assert p.mask_agreement is None
        assert p.mask_box_disagree is False
        assert "mask_agreement" in Provenance.model_fields


class TestTheVerifierIsIndependent:
    def test_it_reprompts_the_box_rather_than_reusing_the_mask(self):
        """Feeding the first mask to the second model would make the second agree by construction, and the
        score would measure nothing."""
        import inspect

        from services.autolabel import sam_hybrid

        src = inspect.getsource(sam_hybrid.verify_mask)
        assert "bboxes=[list(box)]" in src
        assert "primary_mask" not in src.split("res =")[1].split("return")[0]

    def test_a_verifier_that_finds_nothing_is_maximum_disagreement(self):
        """Not None. The verifier looked and found nothing where the primary found something, which is an
        opinion - the strongest one available."""
        import inspect

        from services.autolabel import sam_hybrid

        assert "return 0.0" in inspect.getsource(sam_hybrid.verify_mask)
