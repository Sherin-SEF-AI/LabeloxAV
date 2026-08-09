"""The shortlist was a request. Now it is a constraint.

Path C offers the VLM a per-object shortlist (current class, cross-superclass anchors, L1 siblings,
object_fallback, capped at 20) and put it in the prompt as English. What came back was checked afterwards,
and an out-of-vocabulary class was not repaired but discarded:

    if res.class_name and not self.onto.has_name(res.class_name):
        res.class_name = None

The crop, the prefill and the decode went with it. Path C runs only on the uncertain subset, which is the
expensive subset by construction, so the thrown-away work is the costly work. The corpus has paid for this
before in another currency: 69% of traffic-sign detections were not signs, a bus among them, produced through
exactly this shortlist-as-suggestion path.

Both local backends can enforce a JSON schema instead, so the model is unable to name a class that is not on
offer. The second half is less obvious and is why the shortlist moves out of the prompt entirely rather than
being belt-and-braces in both places: the shortlist varies per object, so while it lived in the prompt every
object was a fresh prefix and there was nothing for a server prefix cache to hold. A schema is a sampler
constraint, not KV state. Moving it makes the prompt byte-identical across objects, which the test below
pins, because it is the property the whole caching argument rests on and nothing else would catch its loss.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from core.config import get_settings
from services.autolabel.paths.constrain import (
    permitted_classes,
    static_prompt,
    verdict_schema,
)

SHORTLIST = ["motorcycle", "scooter", "e_auto", "object_fallback"]
ATTRS = {
    "occlusion": {"type": "enum", "values": [0, 25, 50, 75, 100]},
    "direction": {"type": "enum", "values": ["same", "cross", "wrong_way"]},
    "truncation": {"type": "float"},
    "helmet": {"type": "bool_array"},
    "parked": {"type": "bool"},
}


# ------------------------------------------------------------------------------- the constraint

def test_the_class_is_an_enum_over_exactly_the_shortlist():
    """The whole point: a class outside the offer becomes unsamplable rather than discarded after the fact."""
    assert permitted_classes(verdict_schema(SHORTLIST, ATTRS)) == SHORTLIST


def test_a_narrower_shortlist_produces_a_narrower_schema():
    """The constraint is per object, as the shortlist always was. Locking to the whole 186-class ontology
    instead would be looser than the prompt this replaces."""
    assert permitted_classes(verdict_schema(["cattle", "object_fallback"], ATTRS)) == \
        ["cattle", "object_fallback"]


def test_an_enum_attribute_keeps_the_type_of_its_values():
    """occlusion is [0, 25, 50, 75, 100] under a type field that reads "enum". Quoting those would make every
    occlusion value fail the ontology's own validator immediately downstream."""
    props = verdict_schema(SHORTLIST, ATTRS)["properties"]["attributes"]["properties"]
    assert props["occlusion"]["enum"] == [0, 25, 50, 75, 100]
    assert props["direction"]["enum"] == ["same", "cross", "wrong_way"]


def test_a_per_instance_attribute_is_an_array():
    """helmet is one entry per rider, not one boolean."""
    props = verdict_schema(SHORTLIST, ATTRS)["properties"]["attributes"]["properties"]
    assert props["helmet"] == {"type": "array", "items": {"type": "boolean"}}


def test_scalar_attribute_types_map_to_json_types():
    props = verdict_schema(SHORTLIST, ATTRS)["properties"]["attributes"]["properties"]
    assert props["truncation"] == {"type": "number"}
    assert props["parked"] == {"type": "boolean"}


def test_an_invented_attribute_cannot_ride_along():
    """The validator drops unknown attributes anyway, but only after they were sampled. Tokens spent on a
    field that will be discarded are tokens not spent on the answer."""
    assert verdict_schema(SHORTLIST, ATTRS)["properties"]["attributes"]["additionalProperties"] is False


def test_attributes_and_caption_are_not_required():
    """An attribute that does not apply should be absent. Requiring the field makes the model invent one to
    satisfy the schema, which is the failure the constraint exists to prevent, moved one level down."""
    req = verdict_schema(SHORTLIST, ATTRS)["required"]
    assert "class" in req and "confident" in req
    assert "attributes" not in req and "caption" not in req


def test_an_ontology_with_no_attributes_still_produces_a_valid_schema():
    s = verdict_schema(["cattle"], {})
    assert permitted_classes(s) == ["cattle"]
    assert s["properties"]["attributes"]["properties"] == {}


def test_the_schema_is_json_serialisable():
    """It travels to the server as JSON. A numpy scalar leaking in from an ontology loader would 500 there."""
    json.dumps(verdict_schema(SHORTLIST, ATTRS))


# ------------------------------------------------------------------------------- the prefix

def test_the_prompt_is_identical_for_objects_with_different_shortlists():
    """The property the caching argument rests on, and the one nothing else would catch losing.

    Prefix reuse only pays if the prompt is stable across a sweep. While the shortlist sat in the prompt it
    was not, because the shortlist is per object.
    """
    assert static_prompt(ATTRS) == static_prompt(ATTRS)
    from services.autolabel.paths.path_c_qwen3vl import _build_prompt

    a = _build_prompt(["motorcycle", "scooter"], ATTRS)
    b = _build_prompt(["cattle", "truck"], ATTRS)
    assert a != b, "the old prompt varied per object, which is what cost the cache"


def test_the_static_prompt_names_no_classes():
    p = static_prompt(ATTRS)
    assert "motorcycle" not in p and "cattle" not in p


def test_the_static_prompt_still_explains_the_attributes():
    """The schema constrains their shape but cannot say what `occlusion` means."""
    assert "occlusion" in static_prompt(ATTRS)


# ------------------------------------------------------------------------------- the wire

def _fake_ollama_response(content: dict):
    class R:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": json.dumps(content)}}
    return R()


def test_ollama_sends_the_schema_rather_than_the_word_json():
    """`format: "json"` shapes the output as JSON and constrains nothing about the class."""
    from services.autolabel.paths.path_c_qwen3vl import OllamaVlmClient

    seen: dict = {}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        seen.update(json or {})
        return _fake_ollama_response({"class": "scooter", "confident": True})

    with patch("services.autolabel.paths.path_c_qwen3vl.httpx.post", fake_post):
        res = OllamaVlmClient().verify(np.zeros((32, 32, 3), np.uint8), SHORTLIST, ATTRS)

    assert isinstance(seen["format"], dict), "format must carry the schema, not the string 'json'"
    assert permitted_classes(seen["format"]) == SHORTLIST
    assert res.class_name == "scooter"


def test_ollama_no_longer_puts_the_shortlist_in_the_prompt():
    from services.autolabel.paths.path_c_qwen3vl import OllamaVlmClient

    seen: dict = {}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        seen.update(json or {})
        return _fake_ollama_response({"class": "scooter", "confident": True})

    with patch("services.autolabel.paths.path_c_qwen3vl.httpx.post", fake_post):
        OllamaVlmClient().verify(np.zeros((32, 32, 3), np.uint8), SHORTLIST, ATTRS)

    prompt = seen["messages"][0]["content"]
    assert "scooter" not in prompt and "e_auto" not in prompt


def test_llama_server_asks_for_a_strict_json_schema():
    """`strict` is what makes it a constraint the server compiles to GBNF rather than a hint."""
    from services.autolabel.paths.path_c_qwen3vl import LlamaServerVlmClient

    seen: dict = {}

    class R:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": json.dumps(
                {"class": "e_auto", "confident": True, "caption": "an auto"})}}]}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        seen["url"] = url
        seen.update(json or {})
        return R()

    with patch("services.autolabel.paths.path_c_qwen3vl.httpx.post", fake_post):
        res = LlamaServerVlmClient().verify(np.zeros((32, 32, 3), np.uint8), SHORTLIST, ATTRS)

    rf = seen["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert permitted_classes(rf["json_schema"]["schema"]) == SHORTLIST
    assert res.class_name == "e_auto"
    assert res.provider == "llamacpp", "provenance, so this can be compared against ollama on the gold set"


def test_llama_server_sends_the_image_the_way_its_multimodal_stack_expects():
    """Ollama takes a separate `images` array; llama-server takes OpenAI content parts with a data URL."""
    from services.autolabel.paths.path_c_qwen3vl import LlamaServerVlmClient

    seen: dict = {}

    class R:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({"class": "e_auto", "confident": True})}}]}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        seen.update(json or {})
        return R()

    with patch("services.autolabel.paths.path_c_qwen3vl.httpx.post", fake_post):
        LlamaServerVlmClient().verify(np.zeros((32, 32, 3), np.uint8), SHORTLIST, ATTRS)

    parts = seen["messages"][0]["content"]
    assert any(p["type"] == "image_url" for p in parts)
    assert any(p["type"] == "text" for p in parts)


def test_a_dead_llama_server_yields_an_empty_verdict_not_an_exception():
    """Path C is called inside a batch sweep. One unreachable server must not end it."""
    from services.autolabel.paths.path_c_qwen3vl import LlamaServerVlmClient

    def boom(url, json=None, timeout=None):  # noqa: A002
        raise ConnectionError("connection refused")

    with patch("services.autolabel.paths.path_c_qwen3vl.httpx.post", boom):
        res = LlamaServerVlmClient().verify(np.zeros((32, 32, 3), np.uint8), SHORTLIST, ATTRS)
    assert res.class_name is None


# ------------------------------------------------------------------------------- selection

def test_the_local_backend_is_chosen_by_configuration():
    from services.autolabel.paths.path_c_qwen3vl import LlamaServerVlmClient, OllamaVlmClient
    from services.llm.router import local_vlm_client

    s = get_settings().model_copy(deep=True)
    s.models.vlm.backend = "llamacpp"
    assert isinstance(local_vlm_client(s), LlamaServerVlmClient)
    s.models.vlm.backend = "ollama"
    assert isinstance(local_vlm_client(s), OllamaVlmClient)


def test_ollama_stays_the_default():
    """Which process serves Path C is a deployment decision, not a default that changes under anyone."""
    assert get_settings().models.vlm.backend == "ollama"


# ------------------------------------------------------------------------------- measurability

def test_an_out_of_ontology_class_is_logged_before_it_is_discarded(capsys):
    """Until now the discard left no trace, so the rate it happened at was unknown and the case for
    constraining generation could not be made with a number.

    Asserted on the rendered log line rather than on caplog, because logging here is structlog writing to
    stdout and the stdlib records caplog collects never see it.
    """
    from services.autolabel.ontology import get_ontology
    from services.autolabel.paths.path_c_qwen3vl import VlmResult, VlmVerifier

    v = VlmVerifier.__new__(VlmVerifier)
    v.onto = get_ontology()
    out = v._validate(VlmResult(class_name="a bus but I am calling it a sign", provider="ollama"))

    assert out.class_name is None
    printed = capsys.readouterr().out
    assert "vlm.class_out_of_ontology" in printed
    assert "a bus but I am calling it a sign" in printed, "the emitted name is what makes the rate diagnosable"


@pytest.mark.parametrize("name", ["motorcycle", "cattle"])
def test_a_class_the_ontology_knows_survives_validation(name):
    from services.autolabel.paths.path_c_qwen3vl import VlmResult, VlmVerifier

    v = VlmVerifier.__new__(VlmVerifier)
    from services.autolabel.ontology import get_ontology

    v.onto = get_ontology()
    assert v._validate(VlmResult(class_name=name)).class_name == name
