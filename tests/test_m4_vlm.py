"""M4 tests: VLM duty-cycling (only the uncertain subset), attribute population and validation,
and a real-Ollama backend smoke test (skipped if Ollama is not serving the vision model)."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from core.config import get_settings
from core.schemas import BBox, GateState, PathProposal, Provenance, UnifiedObject
from services.autolabel.gate import gate_object, needs_vlm
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


def test_duty_cycle_only_uncertain_objects():
    onto = get_ontology()
    cfg = get_settings().gate

    confident = _obj(11, "sedan", 0.97, True, [PathProposal(path="path_a_yolo26", verdict="agree", model_version="y")])
    review_band = _obj(11, "sedan", 0.72, True)
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
        _obj(11, "sedan", 0.72, True),     # review band -> VLM
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
    # still rare -> gate keeps it in review (rare forces review even when confident)
    assert gate_object(o, onto, cfg) == GateState.review


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
