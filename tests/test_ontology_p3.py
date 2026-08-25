"""The P3 India ontology: aliases, the per-class attribute scope, ranges and derived attributes.

Most of these assert on load-time refusals. The ontology is data an operator edits by hand, and the failures
that matter are the quiet ones: an alias that shadows a class name, an attribute misspelled in a scope list,
a derived attribute with nothing deriving it. Each of those produces a working system that is wrong rather
than a system that stops.
"""

from __future__ import annotations

import textwrap

import pytest

from services.autolabel.ontology import get_ontology, load_ontology

BASE = """
version: "test-1.0"
hierarchy_levels: 4
attributes:
  occlusion: {{ type: enum, values: [0, 50] }}
  occupant_count: {{ type: int, range: [0, 6] }}
  triple_riding: {{ type: bool, derived_from: occupant_count }}
  footboard_passengers: {{ type: bool }}
attribute_scope:
  two_wheeler: [occlusion, occupant_count, triple_riding]
  heavy: [occlusion]
{extra}classes:
  - {{ id: 1, name: motorcycle, l0: object, l1: two_wheeler, india: false{m_alias} }}
  - {{ id: 2, name: bus, l0: object, l1: heavy, india: false }}
"""


def write(tmp_path, *, extra: str = "", m_alias: str = ""):
    p = tmp_path / "onto.yaml"
    p.write_text(textwrap.dedent(BASE).format(extra=extra, m_alias=m_alias))
    return p


class TestLoadRefusals:
    def test_alias_shadowing_a_class_name_is_refused(self, tmp_path):
        # Both `bus` entries would resolve, and which one an importer got would depend on iteration order.
        with pytest.raises(ValueError, match="already a class name"):
            load_ontology(write(tmp_path, m_alias=", aliases: [bus]"))

    def test_two_classes_cannot_claim_one_alias(self, tmp_path):
        p = tmp_path / "onto.yaml"
        p.write_text(textwrap.dedent(BASE).format(extra="", m_alias=", aliases: [twowheel]")
                     .replace("id: 2, name: bus, l0: object, l1: heavy, india: false",
                              "id: 2, name: bus, l0: object, l1: heavy, india: false, aliases: [twowheel]"))
        with pytest.raises(ValueError, match="claimed by both"):
            load_ontology(p)

    def test_scope_naming_an_unknown_attribute_is_refused(self, tmp_path):
        # The quiet one: `occlusio` scopes nothing out, it removes `occlusion` from the class silently.
        with pytest.raises(ValueError, match="names unknown attributes"):
            load_ontology(write(tmp_path, extra="attribute_scope_class:\n  bus: [occlusio]\n"))

    def test_class_scope_naming_an_unknown_class_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="names unknown classes"):
            load_ontology(write(tmp_path, extra="attribute_scope_class:\n  lorry: [footboard_passengers]\n"))

    def test_derived_from_an_unknown_attribute_is_refused(self, tmp_path):
        p = tmp_path / "onto.yaml"
        p.write_text(textwrap.dedent(BASE).format(extra="", m_alias="")
                     .replace("derived_from: occupant_count", "derived_from: riders"))
        with pytest.raises(ValueError, match="derives from unknown attribute"):
            load_ontology(p)


class TestClassScopeLayer:
    def test_extras_union_onto_the_l1_list(self, tmp_path):
        o = load_ontology(write(tmp_path, extra="attribute_scope_class:\n  bus: [footboard_passengers]\n"))
        # The bus keeps every heavy attribute and gains one. Override rather than union would have silently
        # dropped `occlusion` from every bus in the corpus.
        assert o.attrs_for_class(2) == ["occlusion", "footboard_passengers"]

    def test_a_class_without_extras_is_untouched(self, tmp_path):
        o = load_ontology(write(tmp_path, extra="attribute_scope_class:\n  bus: [footboard_passengers]\n"))
        assert o.attrs_for_class(1) == ["occlusion", "occupant_count", "triple_riding"]

    def test_the_extra_is_accepted_on_the_scoped_class_and_refused_elsewhere(self, tmp_path):
        o = load_ontology(write(tmp_path, extra="attribute_scope_class:\n  bus: [footboard_passengers]\n"))
        assert o.validate_attrs({"footboard_passengers": True}, 2) == []
        assert o.validate_attrs({"footboard_passengers": True}, 1) != []


class TestRanges:
    def test_both_ends_of_an_int_range_are_enforced(self, tmp_path):
        # The int branch checked the type and never the range, so any out-of-range integer validated.
        o = load_ontology(write(tmp_path))
        assert o.validate_attrs({"occupant_count": 7}, 1) != []
        assert o.validate_attrs({"occupant_count": -1}, 1) != []
        assert o.validate_attrs({"occupant_count": 3}, 1) == []

    def test_a_bool_is_still_not_an_int(self, tmp_path):
        o = load_ontology(write(tmp_path))
        assert o.validate_attrs({"occupant_count": True}, 1) != []


class TestDerived:
    def test_a_derived_attribute_cannot_be_written_directly(self, tmp_path):
        o = load_ontology(write(tmp_path))
        errs = o.validate_attrs({"triple_riding": True}, 1)
        assert errs and "occupant_count" in errs[0]

    def test_it_is_computed_from_its_source(self, tmp_path):
        o = load_ontology(write(tmp_path))
        assert o.derive_attrs({"occupant_count": 3}, 1)["triple_riding"] is True
        assert o.derive_attrs({"occupant_count": 2}, 1)["triple_riding"] is False

    def test_a_correction_to_the_source_moves_it(self, tmp_path):
        o = load_ontology(write(tmp_path))
        # The whole reason it is derived: stored state that says three occupants and not triple riding is
        # unreadable, and there is no rule for which half to believe.
        assert o.derive_attrs({"occupant_count": 1, "triple_riding": True}, 1)["triple_riding"] is False

    def test_a_stale_value_with_no_source_is_dropped(self, tmp_path):
        o = load_ontology(write(tmp_path))
        assert "triple_riding" not in o.derive_attrs({"triple_riding": True}, 1)

    def test_it_is_dropped_on_a_class_that_does_not_admit_it(self, tmp_path):
        o = load_ontology(write(tmp_path))
        assert "triple_riding" not in o.derive_attrs({"occupant_count": 3}, 2)


class TestShippedOntology:
    """Assertions against the real YAML, not a fixture."""

    def test_it_loads_and_is_at_the_bumped_version(self):
        assert get_ontology().version == "labelox-in-0.2.0"

    def test_governed_ids_stay_contiguous_from_one(self):
        ids = sorted(c.id for c in get_ontology().classes if c.id <= 199)
        assert ids == list(range(1, len(ids) + 1))

    def test_the_frozen_ids_still_mean_what_they_meant(self):
        # The gold sets carried forward by migration 0101 are only valid evidence because of this.
        o = get_ontology()
        for cid, name in ((1, "motorcycle"), (6, "autorickshaw"), (16, "bus"), (19, "truck"), (31, "cattle")):
            assert o.by_id(cid).name == name

    def test_every_class_resolves_from_its_own_name(self):
        o = get_ontology()
        assert all(o.aliases_for(c.id)[0] == c.name for c in o.classes)

    def test_the_duplicates_the_brief_asked_for_became_aliases(self):
        # Seven of the twelve requested leaves already existed under other names. Each is reachable by the
        # name the brief used without a second class carrying half the corpus.
        o = get_ontology()
        for word, owner in (("auto_rickshaw", "autorickshaw"), ("share_auto", "shared_auto"),
                            ("handcart", "push_cart"), ("head_load_carrier", "person_carrying_load"),
                            ("jcb", "excavator"), ("vendor_cart", "vendor_handcart")):
            hits = [c.name for c in o.classes if word in c.aliases]
            assert hits == [owner], f"{word} -> {hits}"

    def test_bus_service_is_offered_on_buses_and_not_on_trucks(self):
        o = get_ontology()
        assert "bus_service" in o.attrs_for_class(16)
        assert "bus_service" not in o.attrs_for_class(19)

    def test_the_new_relations_exist_in_the_pack(self):
        from services.domain import active_pack

        assert {"towing", "pulling", "herding"} <= active_pack().relations.kinds


class TestInfrastructureAndHazards:
    """WP3: the surface defects, the writing, and what counts as safety-critical."""

    def test_pothole_and_open_manhole_exist_at_all(self):
        """A 166-class India road ontology had no pothole in it. That is the single most surprising thing
        the audit turned up, and this is the assertion that keeps it true that it does now."""
        o = get_ontology()
        for name in ("pothole", "manhole", "open_manhole", "road_cut"):
            assert o.has_name(name), name

    def test_a_hole_in_the_road_is_safety_critical(self):
        """Every other member of the set is something a vehicle might hit. These two are things a
        two-wheeler falls into, in a country where two-wheelers are most of the traffic."""
        from services.domain import critical_class_ids, critical_class_names

        o = get_ontology()
        assert {"pothole", "open_manhole"} <= critical_class_names()
        # By id as well: services/verdyx/safety_recall.py once carried a 0-based list that disagreed with
        # the governed 1-based ids, so name-to-id has to be exercised rather than assumed.
        assert {o.by_name("pothole").id, o.by_name("open_manhole").id} <= critical_class_ids(o)

    def test_severity_is_an_attribute_and_not_three_classes(self):
        o = get_ontology()
        assert "hazard_severity" in o.attrs_for_class(o.by_name("pothole").id)
        assert not any(n in o._by_name for n in ("severe_pothole", "minor_pothole"))

    def test_road_text_records_the_script_and_never_the_text(self):
        """No OCR model, no text field, and this is where that stays true. A sign's script is what tells you
        whether a detector trained on Latin signage can read this corpus, and that needs no transcription.
        Transcription is also where a road dataset starts collecting shopfronts and number plates."""
        o = get_ontology()
        assert o.has_name("road_text")
        assert "script" in o.attrs_for_class(o.by_name("road_text").id)
        forbidden = {"text", "text_content", "transcription", "ocr_text", "content", "reads"}
        assert not (forbidden & set(o.attributes)), forbidden & set(o.attributes)

    def test_script_is_a_set_because_one_board_carries_two(self):
        o = get_ontology()
        sign = o.by_name("traffic_sign").id
        assert o.validate_attrs({"script": ["kannada", "latin"]}, sign) == []
        assert o.validate_attrs({"script": "latin"}, sign) != []
        assert o.validate_attrs({"script": ["latin", "latin"]}, sign) != []
        assert o.validate_attrs({"script": ["klingon"]}, sign) != []

    def test_an_unimplemented_attribute_type_is_refused_rather_than_waved_through(self, tmp_path):
        """The fall-through used to accept anything, so adding a type to the YAML silently disabled
        validation for every attribute using it."""
        p = tmp_path / "onto.yaml"
        p.write_text(textwrap.dedent(BASE).format(extra="", m_alias="")
                     .replace("occlusion: { type: enum, values: [0, 50] }", "occlusion: { type: colour }"))
        o = load_ontology(p)
        assert o.validate_attrs({"occlusion": "anything at all"}, 1) != []

    def test_what_already_existed_did_not_get_a_second_name(self):
        """`open_drain` is `drain` 152, `kerb` is `curb` 101, `footpath` is `sidewalk` 102. The surface and
        infra space already held 75 classes and most of the brief's list was in it."""
        o = get_ontology()
        for word in ("open_drain", "kerb", "footpath", "tree_branch", "police_barricade",
                     "unmarked_speed_breaker", "broken_down_vehicle"):
            assert not o.has_name(word), f"{word} was added as a class"


class TestTheParserCatchesWhatValidationCannot:
    def test_a_duplicate_key_is_refused_rather_than_silently_taking_the_last(self, tmp_path):
        """`yaml.safe_load` takes the last of a repeated key and says nothing. This file spent a commit with
        `aliases` written twice on seven class lines: it loaded, it validated, every test passed, and the
        file was wrong. Nothing downstream of the parser can catch that, so the parser has to."""
        p = tmp_path / "onto.yaml"
        p.write_text(textwrap.dedent(BASE).format(
            extra="", m_alias=", aliases: [bike], aliases: [twowheel]"))
        with pytest.raises(ValueError, match="duplicate key"):
            load_ontology(p)

    def test_the_shipped_yaml_has_no_duplicate_keys(self):
        from services.autolabel.ontology import load_ontology as _load

        _load()  # raises if any mapping repeats a key
