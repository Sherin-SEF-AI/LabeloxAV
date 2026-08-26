"""SEC-M4: the engine consumes the pack's safety/eval definitions.

Proves the consolidation is byte-identical for AV (the safety predicate, protected slices, strata dims, and
critical ids all resolve to what the engine used before), that the latent bugs are fixed, and that a non-AV
pack (sec) routes to its own definitions when addressed by id.
"""

from __future__ import annotations

from services import domain
from services.autolabel.ontology import get_ontology

# ---- AV: byte-identical to the pre-consolidation literals -----------------------------------------------

def test_av_safety_l1_is_vru_animal():
    assert set(domain.safety_l1()) == {"vru", "animal"}
    assert set(domain.safety_l1("av")) == {"vru", "animal"}


def test_av_safety_predicate_matches_the_ontology_rule():
    onto = get_ontology("av")
    for name in ("pedestrian", "rider", "cattle"):
        assert domain.is_safety_class(name) == (onto.by_name(name).l1 in {"vru", "animal"}) is True
    for name in ("sedan", "truck", "pole"):
        assert domain.is_safety_class(name) is False


def test_av_protected_slices_match_the_old_hardcoded_fallback():
    # verdyx/run.py used to hardcode exactly this; it now flows from the pack.
    assert domain.protected_slices() == ("pedestrian_night", "autorickshaw_glare")


def test_av_strata_dimensions_are_the_scene_axes():
    from services.curation.slices import _SCENE_AXES
    assert domain.strata_dimensions() == tuple(_SCENE_AXES)


def test_verdyx_protected_uses_the_pack_when_config_empty():
    from core.config import get_settings
    from services.verdyx.run import _protected
    # GovernSettings.protected_slices defaults empty -> the pack default flows through.
    assert tuple(_protected(get_settings().phase4.govern)) == domain.protected_slices()


# ---- bug #2: critical classes are ontology-resolved by name, not the 0-based {0,1,2,3,8} ----------------

def test_critical_ids_are_ontology_resolved_not_zero_based():
    onto = get_ontology("av")
    ids = domain.critical_class_ids(onto)
    # The subject is the resolution, not the membership. Deriving the expectation from the pack's own set
    # rather than restating it: a literal list here made adding pothole and open_manhole to the safety set
    # look like a regression in id resolution, which is the one thing this test is not about.
    expected = {onto.by_name(n).id for n in domain.critical_class_names()}
    assert ids == expected
    # 0, which the old 0-based bug included, is not among them, and every id resolves to a real class.
    assert 0 not in ids
    assert all(onto.by_id(i).name in domain.critical_class_names() for i in ids)
    assert 0 not in ids


def test_critical_object_recall_uses_pack_ids_by_default():
    from services.verdyx.safety_recall import critical_object_recall
    onto = get_ontology("av")
    ped, sedan = onto.by_name("pedestrian").id, onto.by_name("sedan").id
    objs = [
        {"class_id": ped, "detected": True},
        {"class_id": ped, "detected": False},
        {"class_id": sedan, "detected": True},   # not critical -> excluded
    ]
    r = critical_object_recall(objs)
    assert r["n_critical"] == 2 and r["recall"] == 0.5


# ---- sec: a non-AV pack routes to its own definitions ---------------------------------------------------

def test_sec_routes_to_its_own_safety_and_strata():
    assert set(domain.safety_l1("sec")) == {"person", "weapon"}
    assert domain.is_safety_class("person", "sec") is True
    assert domain.is_safety_class("weapon", "sec") is True
    assert domain.is_safety_class("car", "sec") is False
    assert domain.strata_dimensions("sec") == ("camera_zone", "time_of_day", "occupancy", "lighting")
    assert domain.protected_slices("sec") == ("night_lowlight", "crowd_occlusion")


def test_sec_critical_ids_resolve_against_the_sec_ontology():
    sec_onto = get_ontology("sec")
    ids = domain.critical_class_ids(sec_onto, "sec")
    expected = {sec_onto.by_name(n).id
                for n in ("person", "weapon", "firearm", "knife", "abandoned_object")}
    assert ids == expected
