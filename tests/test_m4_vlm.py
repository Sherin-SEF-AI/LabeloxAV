"""M4 tests: VLM duty-cycling (only the uncertain subset), attribute population and validation,
and a real-Ollama backend smoke test (skipped if Ollama is not serving the vision model)."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from core.config import get_settings
from core.schemas import BBox, GateState, PathProposal, Provenance, UnifiedObject
from services.autolabel.gate import class_auto_accept, gate_object, needs_vlm
from services.autolabel.ontology import get_ontology
from services.autolabel.paths.path_c_qwen3vl import VlmResult, VlmVerifier, apply_vlm


class FakeVlmClient:
    """Deterministic VLM stand-in. Returns a reclassification + attrs (one invalid, to exercise
    ontology validation)."""

    def __init__(self):
        self.calls = 0

    def verify(self, crop_bgr, shortlist, attr_schema, temperature=0.0) -> VlmResult:
        self.calls += 1
        return VlmResult(
            class_name="autorickshaw",
            attrs={"occlusion": 25, "overload": True, "occlusion_bogus": 33},
            caption="three-wheeler, loaded",
            confident=True,
        )


def _obj(class_id, class_name, conf, agreement, proposals=None) -> UnifiedObject:
    return UnifiedObject(
        frame_id=uuid.uuid4(),
        class_id=class_id,
        class_name=class_name,
        bbox=BBox(x1=50, y1=50, x2=150, y2=150),
        conf=conf,
        provenance=Provenance(agreement=agreement, proposals=proposals or []),
    )


def _review_band_conf(class_id: int, onto, cfg) -> float:
    """A confidence that genuinely sits in the review band for this class, read from the live gate config.

    These tests used to hardcode 0.72, which stopped being a review-band confidence without anybody noticing.
    Calibration moved the gate onto the honest isotonic scale (a14d22c: auto_accept 0.95 -> 0.45), so 0.72
    became an auto-accept, and two tests whose whole subject is "the uncertain subset gets a VLM call" were
    quietly asserting it about an object the gate was confident in. Deriving the value pins them to the
    property they are named for instead of to one calibration epoch, so a future retune cannot silently
    invert their meaning again.
    """
    hi = class_auto_accept(class_id, onto, cfg)
    assert cfg.review_low < hi, (
        f"the review band is empty for class {class_id}: review_low={cfg.review_low} >= auto_accept={hi}. "
        "No confidence can be uncertain, so these tests would assert nothing.")
    return (cfg.review_low + hi) / 2


def test_duty_cycle_only_uncertain_objects():
    onto = get_ontology()
    cfg = get_settings().gate

    confident = _obj(11, "sedan", 0.97, True, [PathProposal(path="path_a_yolo26", verdict="agree", model_version="y")])
    review_band = _obj(11, "sedan", _review_band_conf(11, onto, cfg), True)
    rare = _obj(6, "autorickshaw", 0.98, True)

    assert needs_vlm(confident, onto, cfg) is False  # high-conf agreed common class: skip VLM
    assert needs_vlm(review_band, onto, cfg) is True
    assert needs_vlm(rare, onto, cfg) is True


def test_vlm_runs_on_subset_and_populates_validated_attrs():
    onto = get_ontology()
    cfg = get_settings().gate
    fake = FakeVlmClient()
    verifier = VlmVerifier(fake, onto, get_settings())
    img = np.random.default_rng(0).integers(0, 255, size=(240, 320, 3), dtype=np.uint8)

    objs = [
        _obj(11, "sedan", 0.97, True, [PathProposal(path="path_a_yolo26", verdict="agree", model_version="y")]),
        _obj(11, "sedan", _review_band_conf(11, onto, cfg), True),  # review band -> VLM
        _obj(6, "autorickshaw", 0.98, True),  # rare -> VLM
    ]
    touched = 0
    for o in objs:
        if needs_vlm(o, onto, cfg):
            res = verifier.verify_object(img, tuple(o.bbox.as_list()), o.class_id)
            apply_vlm(o, res, onto, "qwen2.5vl:7b")
            touched += 1

    assert fake.calls == 2  # only the two uncertain objects
    assert touched == 2
    # invalid attribute dropped, valid ones kept
    assert objs[1].attrs.get("overload") is True
    assert "occlusion_bogus" not in objs[1].attrs
    # VLM reclassified the review-band sedan to autorickshaw and recorded a path_c proposal
    assert objs[1].class_name == "autorickshaw"
    assert any(p.path == "path_c_qwen3vl" for p in objs[1].provenance.proposals)


def test_vlm_confirm_boosts_then_regate():
    onto = get_ontology()
    cfg = get_settings().gate

    class ConfirmClient:
        def verify(self, *a, **k):
            return VlmResult(class_name="autorickshaw", attrs={"overload": True}, caption="", confident=True)

    o = _obj(6, "autorickshaw", 0.88, True)
    res = VlmVerifier(ConfirmClient(), onto, get_settings()).verify_object(
        np.zeros((100, 100, 3), np.uint8), tuple(o.bbox.as_list()), o.class_id
    )
    apply_vlm(o, res, onto, "qwen2.5vl:7b")
    assert o.conf >= 0.88  # confirm boosts confidence
    # M-Q.4: a rare class (autorickshaw) now earns auto-accept once it has cross-path agreement AND a VLM
    # confirmation and clears its calibrated threshold; the VLM confirm supplies exactly that.
    assert any(p.path == "path_c_qwen3vl" and p.verdict == "confirm" for p in o.provenance.proposals)
    assert gate_object(o, onto, cfg) == GateState.auto_accept
    # but with the strict escape hatch on, a rare class still never auto-accepts
    strict = cfg.model_copy(update={"force_review_on_rare": True})
    assert gate_object(o, onto, strict) == GateState.review


# --- real Ollama backend smoke test ------------------------------------------


def _ollama_vision_ready() -> bool:
    try:
        import httpx

        tag = get_settings().models.vlm.ollama_tag
        r = httpx.get(f"{get_settings().models.vlm.ollama_url}/api/tags", timeout=3)
        return any(m["name"] == tag for m in r.json().get("models", []))
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_vision_ready(), reason="ollama vision model not available")
def test_ollama_backend_returns_structured_result():
    from services.autolabel.paths.path_c_qwen3vl import OllamaVlmClient

    # A simple synthetic crop; we assert the JSON round-trip works and validates, not the label.
    img = np.full((128, 128, 3), 200, dtype=np.uint8)
    client = OllamaVlmClient(get_settings())
    res = client.verify(img, ["autorickshaw", "sedan", "object_fallback"], {"overload": {"type": "bool"}})
    assert isinstance(res, VlmResult)
    assert isinstance(res.attrs, dict)


class _WholeOntologyVlm:
    """A VLM that answers every attribute it is asked about, which is what a real one does.

    Asked for a traffic signal's state while looking at an autorickshaw, a model does not reply "not
    applicable". This stand-in returns whatever the schema offers, so the test measures what the schema
    asked for rather than what a particular model happened to volunteer.
    """

    def __init__(self):
        self.schemas: list[dict] = []

    def verify(self, crop_bgr, shortlist, attr_schema, temperature=0.0) -> VlmResult:
        self.schemas.append(attr_schema)
        sample = {"enum": lambda a: (a.get("values") or ["x"])[0], "bool": lambda a: False,
                  "int": lambda a: 1, "float": lambda a: 0.1, "bool_array": lambda a: [False]}
        return VlmResult(
            class_name="autorickshaw",
            attrs={k: sample.get(spec["type"], lambda a: "x")(spec) for k, spec in attr_schema.items()},
            caption="", confident=True,
        )


class TestAnAttributeThatCannotApply:
    """7,500 objects in this corpus that are not signals carry a `signal_state`.

    One autorickshaw held signal_state, signal_kind, signal_mount, signal_arrow, marking_state, articulated
    and helmet at once. None of those are observations of anything: the schema handed the model every
    attribute in the ontology on every crop, and `validate_attrs` was called without a class id, which is the
    argument that turns its applicability check on. The values were checked and the applicability never was.
    """

    def test_the_model_is_not_asked_about_attributes_the_class_cannot_have(self):
        onto = get_ontology()
        fake = _WholeOntologyVlm()
        verifier = VlmVerifier(fake, onto, get_settings())
        img = np.zeros((240, 320, 3), dtype=np.uint8)

        verifier.verify_object(img, (50, 50, 150, 150), onto.by_name("autorickshaw").id)

        assert fake.schemas, "the verifier never called the model"
        asked = set(fake.schemas[0])
        for cannot in ("signal_state", "signal_kind", "signal_mount", "signal_arrow", "marking_state"):
            assert cannot not in asked, (
                f"the model was asked for '{cannot}' while looking at a three-wheeler; a model asked for a "
                f"fact that cannot apply invents one")
        # The ones a three-wheeler really does carry are still asked for, so this is a scope and not a mute.
        assert {"occlusion", "motion", "passenger_load", "livery"} <= asked

    def test_an_inapplicable_attribute_never_reaches_the_object(self):
        """Belt and braces: even if a model volunteers one unasked, it does not land."""
        onto = get_ontology()
        obj = _obj(onto.by_name("autorickshaw").id, "autorickshaw", 0.9, True)
        res = VlmResult(class_name="autorickshaw", confident=True,
                        attrs={"signal_state": "R", "articulated": True, "helmet": [False],
                               "marking_state": "present", "motion": "moving", "livery": True})
        apply_vlm(obj, res, onto, "test")

        for cannot in ("signal_state", "articulated", "helmet", "marking_state"):
            assert cannot not in obj.attrs, f"'{cannot}' landed on a three-wheeler"
        assert obj.attrs["motion"] == "moving" and obj.attrs["livery"] is True

    def test_a_reclassification_keeps_the_attributes_of_the_class_it_moved_to(self):
        """The ordering that matters. Filtering before the reclassification judges the reply against the
        class the object is leaving, which drops exactly the attributes the new class does carry."""
        onto = get_ontology()
        obj = _obj(onto.by_name("sedan").id, "sedan", 0.9, True)
        res = VlmResult(class_name="autorickshaw", confident=True, votes=3, agreement=1.0,
                        attrs={"overload": True, "passenger_load": 2})
        apply_vlm(obj, res, onto, "test")

        assert obj.class_name == "autorickshaw"
        assert obj.attrs.get("overload") is True, "filtered against the class it was leaving"
        assert obj.attrs.get("passenger_load") == 2
