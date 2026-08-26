"""SEC-M1: the pack scaffolding + AV byte-parity.

These prove (a) the registry discovers and validates the AV pack, (b) get_ontology stays byte-identical for
the AV default, and (c) the AV pack faithfully mirrors the engine values it is authored from (the parity that
the later milestones rely on when they point engine call sites at the pack).
"""

from __future__ import annotations

import pytest
import yaml

from packs.base import DomainPack
from packs.registry import PACKS_DIR, get_pack, pack_ids
from services.autolabel.ontology import get_ontology, load_ontology


def test_registry_discovers_av():
    assert "av" in pack_ids()


def test_av_is_a_domainpack():
    pack = get_pack("av")
    assert isinstance(pack, DomainPack)


def test_unknown_pack_raises_with_known_ids():
    with pytest.raises(KeyError) as ei:
        get_pack("does_not_exist")
    assert "av" in str(ei.value)


def test_code_manifest_matches_pack_yaml():
    pack = get_pack("av")
    meta = yaml.safe_load((PACKS_DIR / "av" / "pack.yaml").read_text())
    assert meta["id"] == pack.manifest.id
    assert meta["version"] == pack.manifest.version
    assert meta["default_scene_model"] == pack.manifest.default_scene_model
    assert set(meta["capabilities"]) == set(pack.manifest.capabilities)


# ---- get_ontology parity -------------------------------------------------------------------------------

def test_get_ontology_default_is_av():
    # Every historical no-arg caller must resolve to the same object as the explicit AV pack id.
    assert get_ontology() is get_ontology("av")


def test_get_ontology_av_is_byte_identical():
    a = get_ontology("av")
    fresh = load_ontology()  # the pre-pack code path
    assert a.version == fresh.version
    assert [(c.id, c.name, c.l0, c.l1, c.india) for c in a.classes] == \
           [(c.id, c.name, c.l0, c.l1, c.india) for c in fresh.classes]


# ---- AV pack mirrors the engine values it was authored from --------------------------------------------

def test_safety_l1_matches_the_affinity_source():
    """SEC-M4 pointed the engine's safety predicates at the pack (champion/recall/autolabel/dynamics no longer
    keep a `{"vru","animal"}` literal). safe_miou remains the AV affinity implementation the pack delegates to,
    so its l1 set must still equal the pack's - the one place the two definitions must not drift."""
    from services.domain import safety_l1
    from services.training.safe_miou import _VRU_ANIMAL as miou

    pack_set = set(get_pack("av").autolabel_profile.gate_policy.safety_l1)
    assert pack_set == set(miou) == set(safety_l1())


def test_safety_policy_affinity_delegates_to_safe_miou():
    from services.training.safe_miou import affinity_cost

    pack = get_pack("av")
    onto = get_ontology("av")
    ped, sedan = onto.by_name("pedestrian").id, onto.by_name("sedan").id
    assert pack.safety_policy.affinity_cost(ped, sedan) == affinity_cost(onto, ped, sedan)
    assert pack.safety_policy.affinity_cost(ped, ped) == 0.0


def test_critical_class_names_are_governed():
    pack, onto = get_pack("av"), get_ontology("av")
    for name in pack.safety_policy.critical_class_names():
        assert onto.has_name(name), f"critical class {name} missing from ontology"


def test_vlm_prompt_template_is_the_live_prompt_preamble():
    """The pack's prompt template must still be the preamble path_c actually emits (the SEC-M5 seam)."""
    from services.autolabel.paths.path_c_vlm import _build_prompt

    live = _build_prompt(["sedan", "pedestrian"], {})
    assert live.startswith(get_pack("av").autolabel_profile.vlm_prompt_template)


def test_protected_slices_match_verdyx_default():
    from services.verdyx.run import _protected

    # A cfg without the (still-missing) protected_slices attr falls back to the AV default in verdyx/run.py.
    assert tuple(_protected(object())) == get_pack("av").eval_strata.protected_slices


def test_forge_targets_match_the_silicon_registries():
    from services.forgyx.cooptimize import TARGET_BUDGET_MS
    from services.forgyx.thermal import TARGET_THERMAL

    targets = {t.name: t for t in get_pack("av").forge_targets}
    assert set(targets) == set(TARGET_BUDGET_MS)
    for name, t in targets.items():
        assert t.latency_budget_ms == TARGET_BUDGET_MS[name]
        assert t.throttle_temp_c == TARGET_THERMAL[name]["throttle_temp_c"]
        assert t.power_ceiling_w == TARGET_THERMAL[name]["power_ceiling_w"]


def test_ontology_spec_mirrors_engine_constants():
    from services.autolabel.ontology import CUSTOM_ID_BASE, STUFF_L0, STUFF_NAMES

    spec = get_pack("av").ontology
    assert spec.stuff_names == STUFF_NAMES
    assert spec.stuff_l0 == STUFF_L0
    assert spec.custom_id_base == CUSTOM_ID_BASE
