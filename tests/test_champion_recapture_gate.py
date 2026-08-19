"""The promotion gate, against a gold denominator it now knows might be inflated.

Every floor in champion_gate (mAP uplift, safety-class AP, safety recall) is computed against the sealed
gold set, and that set was built by people who confirm machine boxes far more readily than they draw new
ones. A challenger can therefore clear every floor while being worse at the thing the floors protect,
because the floors and the challenger share the same blind spot.

These tests pin the four states the recapture condition can be in, and the last two fail against the gate
as it stood before this change: it had no recapture key at all, so an overstated denominator promoted.

Pure: champion_gate is a pure function over two metric dicts, so none of this needs a database.
"""

from __future__ import annotations

from types import SimpleNamespace

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.govern.champion import champion_gate

CHAMP = {"map": 0.70, "safe_miou": 0.90,
         "per_class": {"pedestrian": 0.80, "child": 0.78, "motorcycle": 0.65},
         "per_class_recall": {"pedestrian": 0.80, "child": 0.78, "motorcycle": 0.65}}

# A challenger that clears every pre-existing floor: better mAP, no Safe-mIoU drop, no class regression.
CHAL = {"map": 0.74, "safe_miou": 0.91,
        "per_class": {"pedestrian": 0.82, "child": 0.79, "motorcycle": 0.70},
        "per_class_recall": {"pedestrian": 0.82, "child": 0.79, "motorcycle": 0.70}}


def _cfg(**over):
    """The real govern settings with named fields overridden, so a default change shows up here."""
    base = get_settings().phase4.govern
    fields = {k: getattr(base, k) for k in base.model_fields}
    fields.update(over)
    return SimpleNamespace(**fields)


def _est(**over):
    est = {"measured": True, "gold_recall": 0.80, "model_recall": 0.78,
           "n_both": 120, "n_model_only": 40, "n_human_only": 45, "reason": None}
    est.update(over)
    return est


class TestUnchecked:
    def test_no_audit_is_advisory_by_default_and_says_so(self):
        """Off by default, and the reason is on every promotion rather than nowhere.

        A gate that starts red on every promotion gets switched off inside a week, and there are no blind
        audits yet because one cannot exist until a person labels two hundred frames. Advisory keeps the
        gap visible until it can be closed.
        """
        onto = get_ontology()
        g = champion_gate(CHAL, CHAMP, onto, _cfg(blind_audit_required=False))
        assert g["promote"] is True
        assert g["recapture_ok"] is True
        assert g["recapture"]["status"] == "unchecked"
        assert any("no blind audit" in r for r in g["reasons"]), g["reasons"]
        assert any("advisory" in r for r in g["reasons"])

    def test_no_audit_refuses_once_enforcement_is_switched_on(self):
        onto = get_ontology()
        g = champion_gate(CHAL, CHAMP, onto, _cfg(blind_audit_required=True))
        assert g["promote"] is False
        assert g["recapture_ok"] is False
        assert any("blind_audit_required" in r for r in g["reasons"]), g["reasons"]

    def test_a_first_champion_is_held_to_the_same_condition(self):
        """The baseline is where an unverified denominator does the most damage.

        Every later comparison is made against the first champion, so a bias admitted here is invisible to
        everything downstream: it becomes the thing "no regression" is measured from.
        """
        onto = get_ontology()
        assert champion_gate(CHAL, None, onto, _cfg(blind_audit_required=True))["promote"] is False
        assert champion_gate(CHAL, None, onto, _cfg(blind_audit_required=False))["promote"] is True


class TestMeasured:
    def test_an_overstated_denominator_blocks_a_challenger_that_clears_every_other_floor(self):
        """This is the whole point, and it fails against the gate as it stood.

        Gold says 0.80 recall; the blind audit says 0.55. The challenger still beats the champion on mAP,
        holds Safe-mIoU and regresses no class, so before this change it promoted. It should not: the 0.25
        gap means the gold denominator is missing a quarter of the objects, and every floor above was
        computed on it.
        """
        onto = get_ontology()
        chal = {**CHAL, "recapture": _est(gold_recall=0.80, model_recall=0.55)}
        g = champion_gate(chal, CHAMP, onto, _cfg(recapture_max_overstatement=0.15))
        assert g["promote"] is False
        assert g["recapture_ok"] is False
        assert g["recapture"]["overstatement"] == 0.25
        assert any("overstates measured recall by 0.250" in r for r in g["reasons"]), g["reasons"]
        # The counts travel with the refusal, so the reader can see how much evidence it rests on.
        assert any("120 shared" in r for r in g["reasons"])

    def test_a_small_gap_promotes_and_reports_the_gap(self):
        onto = get_ontology()
        chal = {**CHAL, "recapture": _est(gold_recall=0.80, model_recall=0.78)}
        g = champion_gate(chal, CHAMP, onto, _cfg(recapture_max_overstatement=0.15))
        assert g["promote"] is True
        assert g["recapture"]["status"] == "measured"
        assert abs(g["recapture"]["overstatement"] - 0.02) < 1e-9

    def test_the_tolerance_is_the_thing_that_decides(self):
        onto = get_ontology()
        chal = {**CHAL, "recapture": _est(gold_recall=0.80, model_recall=0.60)}
        assert champion_gate(chal, CHAMP, onto, _cfg(recapture_max_overstatement=0.25))["promote"] is True
        assert champion_gate(chal, CHAMP, onto, _cfg(recapture_max_overstatement=0.15))["promote"] is False

    def test_a_model_better_than_gold_says_never_blocks(self):
        # Negative overstatement: the audit found the model catching things gold never recorded. That is a
        # pleasant surprise, not a fault, and it must not read as one.
        onto = get_ontology()
        chal = {**CHAL, "recapture": _est(gold_recall=0.70, model_recall=0.82)}
        g = champion_gate(chal, CHAMP, onto, _cfg(recapture_max_overstatement=0.15))
        assert g["promote"] is True
        assert g["recapture"]["overstatement"] < 0


class TestUnmeasured:
    def test_an_audit_that_could_not_conclude_is_never_a_pass(self):
        """"We looked and cannot tell" must not read the same as "we looked and it was fine".

        This is the m2 = 0 case: no object was found by both observers, so the population is unbounded
        above. It also fails closed regardless of blind_audit_required, because unlike "unchecked" it is
        not a deployment that has yet to opt in, it is a check that ran and came back empty.
        """
        onto = get_ontology()
        chal = {**CHAL, "recapture": _est(measured=False, gold_recall=None, model_recall=None,
                                          reason="no object was found by both observers")}
        for required in (True, False):
            g = champion_gate(chal, CHAMP, onto, _cfg(blind_audit_required=required))
            assert g["promote"] is False
            assert g["recapture"]["status"] == "unmeasured"
            assert any("could not conclude" in r for r in g["reasons"]), g["reasons"]

    def test_a_measured_flag_without_the_numbers_is_still_refused(self):
        onto = get_ontology()
        chal = {**CHAL, "recapture": _est(gold_recall=None)}
        g = champion_gate(chal, CHAMP, onto, _cfg())
        assert g["promote"] is False and g["recapture"]["status"] == "unmeasured"


def test_the_pre_existing_floors_still_dominate():
    """A safety regression is not rescued by a clean audit.

    Adding a condition to a fail-closed AND must not create a path where a new pass unblocks an old fail.
    """
    onto = get_ontology()
    regressed = {**CHAL, "per_class": {**CHAL["per_class"], "pedestrian": 0.10},
                 "recapture": _est(gold_recall=0.80, model_recall=0.80)}
    g = champion_gate(regressed, CHAMP, onto, _cfg())
    assert g["promote"] is False
    assert g["recapture_ok"] is True and g["safety_ok"] is False


def test_a_refused_run_reports_the_key_rather_than_omitting_it():
    # Both early refusals return before the recapture check runs. They must still carry the key, or a
    # caller reading gate["recapture_ok"] gets a KeyError on exactly the paths that already went wrong.
    onto = get_ontology()
    for bad in ({"reconstructed": True}, {"harness_divergent": True}):
        g = champion_gate({**CHAL, **bad}, CHAMP, onto, _cfg())
        assert g["promote"] is False and g["recapture_ok"] is False
