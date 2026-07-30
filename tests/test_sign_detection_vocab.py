"""The detector has to be able to propose "advertising" and "road sign" as different things.

Measured on the reviewed corpus: `traffic_sign` holds 48,322 objects against 518 for `hoarding`, and a
sample of its training crops contains petrol price boards, an ice cream hoarding, an Amaron battery advert
and several shop name boards alongside the STOP signs. Recall on the class is 0.163 despite it having the
fourth largest instance count, while classes with a coherent visual definition and similar counts reach
0.68 to 0.74.

Two causes, both in the vocabulary rather than the model:

  1. `hoarding` was stuff, so `persist.py` dropped every one the autolabeller proposed. Advertising had
     nowhere to go, and the nearest surviving class was `traffic_sign`.
  2. The only open-vocabulary phrase for signs was "traffic sign", produced by replacing the underscore in
     the class name. One generic noun phrase had to cover 21 distinct Indian sign designs.
"""

from __future__ import annotations

from services.autolabel.ontology import get_ontology


def test_a_hoarding_is_a_countable_thing():
    """A billboard has definite edges and can be counted, which is what separates a thing from stuff.

    It sat in STUFF_NAMES beside walls and tree canopies. The consequence was not philosophical:
    `services/autolabel/persist.py` keeps only things, so proposals for advertising were discarded and the
    contents ended up under `traffic_sign`.
    """
    onto = get_ontology()
    assert onto.is_thing(onto.by_name("hoarding").id), (
        "hoarding is stuff, so autolabel drops it and advertising lands in traffic_sign instead")


def test_hoarding_is_offered_to_the_open_vocabulary_path():
    """Being a thing is not enough; a class nothing ever proposes is still never labelled."""
    from packs.av.pack import PACK

    assert "hoarding" in PACK.ontology.supported_core


def test_signs_have_more_than_one_phrase_to_match_on():
    """"traffic sign" alone had to cover 21 designs, from an octagonal STOP to a green destination board."""
    from packs.av.pack import PACK

    syn = PACK.autolabel_profile.openvocab_synonyms
    phrases = syn.get("traffic_sign") or ()
    assert len(phrases) >= 5, f"traffic_sign has only {len(phrases)} open-vocab phrase(s)"
    joined = " ".join(phrases).lower()
    for shape in ("road sign", "warning", "speed limit"):
        assert shape in joined, f"no open-vocab phrase covers {shape!r}"


def test_advertising_phrases_do_not_mention_traffic_signs():
    """If the hoarding prompts say "sign", the two classes compete for the same detections."""
    from packs.av.pack import PACK

    syn = PACK.autolabel_profile.openvocab_synonyms
    for phrase in syn.get("hoarding") or ():
        assert "traffic sign" not in phrase.lower()
        assert "road sign" not in phrase.lower()


def test_sign_phrases_do_not_claim_advertising():
    """And the reverse, so a billboard is not pulled back into traffic_sign by its own prompt."""
    from packs.av.pack import PACK

    syn = PACK.autolabel_profile.openvocab_synonyms
    for phrase in syn.get("traffic_sign") or ():
        for banned in ("billboard", "hoarding", "advertis"):
            assert banned not in phrase.lower(), f"sign phrase {phrase!r} also describes advertising"


def test_mapillary_keeps_the_back_of_a_sign_apart_from_the_front():
    """The blank rear of a sign is not a road sign and must not be imported as one.

    Vistas distinguishes front from back. Collapsing both into `traffic_sign` taught the detector that a
    plain grey rectangle is a sign, and handed the type classifier crops with no sign face to read.
    """
    from services.imports.adapter_mapillary import MAPILLARY_TO_ONTOLOGY

    front = MAPILLARY_TO_ONTOLOGY.get("object--traffic-sign--front")
    back = MAPILLARY_TO_ONTOLOGY.get("object--traffic-sign--back")
    assert front == "traffic_sign"
    assert back != "traffic_sign", "the back of a sign is still imported as a traffic sign"
