"""Recall recovery, pure tier: geometry, mining, fusion, scoring, and the recall promotion gate. No
infrastructure, no torch (the model backends are imported lazily only inside run_recall and the adapter
methods, never here). The ontology is the real governed one, like tests/test_m44_govern.py.
"""

from __future__ import annotations

from core.config import get_settings
from services.autolabel.ontology import get_ontology
from services.govern.champion import champion_gate
from services.recall.gate import resolve_safety_drop, safety_recall_floor, safety_recall_no_regress
from services.recall.recover import (
    FusedRecall,
    RecallProposal,
    fn_value,
    fuse_channels,
    interp_box,
    iou_matrix,
    mine_unmatched,
    track_gap_proposals,
)

ONTO = get_ontology()
RCFG = get_settings().phase4.recall
GCFG = get_settings().phase4.govern

VRU = next(c.name for c in ONTO.classes if c.l1 == "vru")
ANIMAL = next((c.name for c in ONTO.classes if c.l1 == "animal"), None)
NONSAFE = next(c.name for c in ONTO.classes if c.l1 not in ("vru", "animal"))


# ---- geometry --------------------------------------------------------------------------------------

def test_iou_matrix():
    a = [[0, 0, 10, 10]]
    assert iou_matrix(a, a)[0, 0] == 1.0                     # identical
    m = iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])       # half overlap: inter 50, union 150
    assert abs(float(m[0, 0]) - (50.0 / 150.0)) < 1e-9
    assert float(iou_matrix([[0, 0, 10, 10]], [[20, 20, 30, 30]])[0, 0]) == 0.0  # disjoint
    assert iou_matrix([], [[0, 0, 1, 1]]).shape == (0, 1)    # empty inputs, correct shape
    assert iou_matrix([], []).shape == (0, 0)


def test_interp_box():
    assert interp_box([0, 0, 10, 10], [100, 0, 110, 10], 0.5) == (50.0, 0.0, 60.0, 10.0)  # midpoint
    assert interp_box([0, 0, 10, 10], [100, 0, 110, 10], 0.0) == (0.0, 0.0, 10.0, 10.0)   # endpoint a
    assert interp_box([0, 0, 10, 10], [100, 0, 110, 10], 1.0) == (100.0, 0.0, 110.0, 10.0)  # endpoint b


# ---- trackgap channel ------------------------------------------------------------------------------

def test_track_gap_proposals_interpolates_interior():
    observed = [(0, [0, 0, 10, 10], 0.9), (100, [100, 0, 110, 10], 0.8)]
    out = track_gap_proposals("trk", 7, observed, [(50, "fA")])
    assert len(out) == 1
    fid, prop = out[0]
    assert fid == "fA"
    assert abs(prop.bbox[0] - 50.0) < 1e-9 and prop.bbox[2] == 60.0  # interpolated x
    assert prop.interp_source == "recall_trackgap" and prop.class_id == 7
    assert prop.channel == "trackgap" and prop.track_id == "trk"


def test_track_gap_proposals_skips_long_gap():
    observed = [(0, [0, 0, 10, 10], 0.9), (100, [100, 0, 110, 10], 0.8)]
    gap = [(t, f"f{t}") for t in (10, 20, 30, 40, 50, 60, 70)]  # 7 interior frames, max is 5
    assert track_gap_proposals("trk", 7, observed, gap, max_gap_frames=5) == []


# ---- mining ----------------------------------------------------------------------------------------

def test_mine_unmatched():
    props = [RecallProposal((0, 0, 10, 10), "region", 0.5), RecallProposal((100, 100, 110, 110), "region", 0.5)]
    existing = [[0, 0, 10, 10]]                              # covers the first proposal
    kept = mine_unmatched(props, existing, iou_match=0.45)
    assert len(kept) == 1 and kept[0].bbox == (100, 100, 110, 110)  # covered one dropped
    assert len(mine_unmatched(props, [], iou_match=0.45)) == 2      # empty existing drops nothing


# ---- fusion ----------------------------------------------------------------------------------------

def test_fuse_channels_collapses_and_inherits():
    far = RecallProposal((100, 100, 110, 110), "openvocab", 0.6, class_name=NONSAFE)
    p_track = RecallProposal((0, 0, 10, 10), "trackgap", 0.7, class_id=7)
    p_open = RecallProposal((1, 1, 11, 11), "openvocab", 0.6, class_name=NONSAFE)  # overlaps p_track
    fused = fuse_channels([p_open, p_track, far], fuse_iou=0.55)
    assert len(fused) == 2                                   # the far box stays separate
    near = next(f for f in fused if f.bbox[0] < 50)
    assert near.channels == {"trackgap", "openvocab"}        # both channels merged onto one candidate
    assert near.class_id == 7                                # trackgap (top rank) kept; its class held

    # a class-less region inherits the named class from a suppressed overlapping proposal
    classless = RecallProposal((0, 0, 10, 10), "region", 0.6, class_name=None)
    named = RecallProposal((1, 1, 11, 11), "region", 0.5, class_name=ANIMAL or NONSAFE)
    fused2 = fuse_channels([classless, named], fuse_iou=0.55)
    assert len(fused2) == 1 and fused2[0].class_name == (ANIMAL or NONSAFE)


# ---- scoring ---------------------------------------------------------------------------------------

def test_fn_value_ranges_and_ranking():
    track_only = FusedRecall((0, 0, 10, 10), {"trackgap"}, 0.7)
    region_only = FusedRecall((0, 0, 10, 10), {"region"}, 0.4)
    assert fn_value(track_only, False, False) > fn_value(region_only, False, False)  # trackgap outranks region
    plain = fn_value(track_only, False, False)
    assert fn_value(track_only, True, False) > plain                                 # rarity raises
    multi = FusedRecall((0, 0, 10, 10), {"trackgap", "openvocab"}, 0.7)
    assert fn_value(multi, False, False) > plain                                     # agreement raises
    assert fn_value(multi, True, True) <= 1.0                                        # clamped


# ---- recall promotion gate -------------------------------------------------------------------------

def test_resolve_safety_drop():
    assert resolve_safety_drop(VRU, ONTO, RCFG) == 0.10          # VRU held tighter
    assert resolve_safety_drop(NONSAFE, ONTO, RCFG) == 0.15      # non-safety gets the default

    class _Scalar:
        safety_class_drop = 0.2
    assert resolve_safety_drop(VRU, ONTO, _Scalar()) == 0.2      # a scalar config is honored


def test_safety_recall_floor():
    healthy = {"per_class_recall": {VRU: 0.80}}
    assert safety_recall_floor(healthy, ONTO, RCFG)["ok"] is True
    blind = {"per_class_recall": {VRU: 0.30}}                    # below the 0.50 floor
    r = safety_recall_floor(blind, ONTO, RCFG)
    assert r["ok"] is False and VRU in r["below_floor"]
    assert safety_recall_floor({"map": 0.9}, ONTO, RCFG)["ok"] is False  # no per_class_recall, refused


def test_safety_recall_no_regress():
    champ = {"per_class_recall": {VRU: 0.80}}
    ok = {"per_class_recall": {VRU: 0.78}}                       # 0.02 drop, within the 0.05 tolerance
    assert safety_recall_no_regress(ok, champ, ONTO, RCFG)["ok"] is True
    bad = {"per_class_recall": {VRU: 0.60}}                      # 0.20 drop, beyond tolerance
    rr = safety_recall_no_regress(bad, champ, ONTO, RCFG)
    assert rr["ok"] is False and VRU in rr["regressed"]


def test_champion_gate_enforces_recall_floor():
    champ = {"map": 0.70, "safe_miou": 0.90, "per_class": {VRU: 0.80}, "per_class_recall": {VRU: 0.80}}
    # improves overall mAP and Safe-mIoU, no AP regression, BUT recall on the safety class collapses
    blind = {"map": 0.85, "safe_miou": 0.92, "per_class": {VRU: 0.82}, "per_class_recall": {VRU: 0.30}}
    assert champion_gate(blind, champ, ONTO, GCFG)["promote"] is False
    # a challenger that reports no per_class_recall at all is refused (fail-closed)
    no_recall = {"map": 0.85, "safe_miou": 0.92, "per_class": {VRU: 0.82}}
    assert champion_gate(no_recall, champ, ONTO, GCFG)["promote"] is False
    # a healthy challenger that also clears the recall floor still promotes
    good = {"map": 0.74, "safe_miou": 0.91, "per_class": {VRU: 0.82}, "per_class_recall": {VRU: 0.82}}
    assert champion_gate(good, champ, ONTO, GCFG)["promote"] is True
