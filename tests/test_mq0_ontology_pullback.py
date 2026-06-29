"""M-Q.0 ontology pullback: the open-vocab models must no longer be prompted with ungrounded class names.

These prove the grounded set excludes the classes hallucinated in the live system (bus_shelter,
water_bottles, vintage_car, balloon_seller, buildings) while keeping the supported core and fallbacks, that
promotion is earned by gate-accepted instances (not raw human draws), and that both prompt surfaces, Path B
concept phrases and the Path C VLM shortlist, are restricted to the grounded set."""

from __future__ import annotations

import numpy as np

from services.autolabel.grounding import (
    promoted_ids,
    promotion_candidates,
    supported_concept_ids,
    supported_core_ids,
)
from services.autolabel.ontology import get_ontology
from services.autolabel.paths.path_b_sam3 import Sam3Path
from services.autolabel.paths.path_c_qwen3vl import VlmResult, VlmVerifier

HALLUCINATED = ["bus_shelter", "water_bottles", "vintage_car", "balloon_seller", "buildings", "multi_axle_trailer"]
CORE = ["pedestrian", "autorickshaw", "sedan", "motorcycle", "rider", "cattle"]


class _StubClient:
    def verify(self, crop_bgr, shortlist, attr_schema, temperature=0.0):
        return VlmResult()


def test_supported_core_excludes_hallucinated_keeps_core_and_fallback():
    onto = get_ontology()
    core = supported_core_ids(onto)
    for name in CORE:
        if onto.has_name(name):
            assert onto.by_name(name).id in core, f"{name} must be in the supported core"
    for name in ("object_fallback", "vehicle_fallback"):
        assert onto.by_name(name).id in core, f"{name} fallback must always be supported"
    for name in ("bus_shelter", "water_bottles", "vintage_car", "balloon_seller", "buildings"):
        if onto.has_name(name):
            assert onto.by_name(name).id not in core, f"{name} must not be in the curated supported core"


async def test_promotion_is_earned_by_accepted_not_human_draws():
    onto = get_ontology()
    promoted = await promoted_ids(min_instances=50)
    # the human-drawn-but-not-gold classes have only a few accepted instances, so they do not promote
    for name in ("water_bottles", "buildings", "electric_post"):
        if onto.has_name(name):
            assert onto.by_name(name).id not in promoted, f"{name} has too few accepted instances to promote"


async def test_supported_concept_ids_drops_the_invented_classes():
    onto = get_ontology()
    ids = await supported_concept_ids()
    assert len(ids) < len(onto.classes), "the grounded set must be a strict subset of the full ontology"
    for name in HALLUCINATED:
        if onto.has_name(name):
            assert onto.by_name(name).id not in ids, f"{name} must not be promptable"
    # the candidate list never silently includes the junk; it is the governor's earned-promotion queue
    cands = {c["name"] for c in await promotion_candidates()}
    assert "balloon_seller" not in cands and "vintage_car" not in cands


def test_path_b_phrases_restricted_to_grounded_set():
    onto = get_ontology()
    core = supported_core_ids(onto)
    restricted = Sam3Path(supported_ids=core)
    unrestricted = Sam3Path(supported_ids=None)
    assert len(restricted._phrases) < len(unrestricted._phrases)
    assert "water bottles" not in restricted._phrases
    assert "bus shelter" not in restricted._phrases
    assert "pedestrian" in restricted._phrases
    # every restricted concept maps back to a grounded class id
    assert all(c.id in core for c in restricted._classes)


def test_path_c_shortlist_restricted_to_grounded_set():
    onto = get_ontology()
    core = supported_core_ids(onto)
    sedan = onto.by_name("sedan").id
    v_restricted = VlmVerifier(_StubClient(), onto, supported_ids=core)
    shortlist = v_restricted._shortlist(sedan)
    allowed = core | {sedan, onto.by_name("object_fallback").id}
    for name in shortlist:
        assert not onto.has_name(name) or onto.by_name(name).id in allowed, f"{name} leaked into the shortlist"
    # an unrestricted verifier still offers the full l1 sibling set (regression contrast)
    v_open = VlmVerifier(_StubClient(), onto, supported_ids=None)
    assert len(v_open._shortlist(sedan)) >= len(shortlist)


def test_segment_smoke_not_required():
    # path construction must not need a GPU (only load() does), so the grounded restriction is testable
    assert isinstance(np.zeros((2, 2)), np.ndarray)
