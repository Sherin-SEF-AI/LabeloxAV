"""LabeloxAV M13 tests: reconciler features + parity gate, 4D propagation with stable identity and gap
healing, and the label-quality layer (flags, inter-annotator agreement, gold audit catching bad labels)."""

from services.labelox.learned_reconcile import reconcile_features, reconcile_parity
from services.labelox.propagate4d import propagate_track
from services.labelox.quality import annotation_quality, gold_audit, inter_annotator_agreement

BOX = [10.0, 10.0, 50.0, 50.0]


def _path(path, cls, conf=0.8, box=None, **kw):
    return {"path": path, "class_id": cls, "conf": conf, "bbox": box or BOX, **kw}


def test_reconcile_features_class_and_box_agreement():
    feats = reconcile_features([_path("a", 6), _path("b", 6), _path("c", 6)])
    assert feats[3] == 1.0            # class agreement
    assert feats[4] > 0.9             # box agreement (identical boxes)
    disagree = reconcile_features([_path("a", 6), _path("b", 11)])
    assert disagree[3] == 0.0


def test_reconcile_parity_gate():
    assert reconcile_parity({"map50": 0.56}, {"map50": 0.55})["promote"] is True
    assert reconcile_parity({"map50": 0.54}, {"map50": 0.55})["promote"] is False
    # with a required margin, a tie does not promote
    assert reconcile_parity({"map50": 0.55}, {"map50": 0.55}, margin=0.01)["promote"] is False


def test_propagate4d_single_identity_and_gap_healing():
    r = propagate_track(BOX, n_frames=6, velocity=[1, 0, 1, 0], known={4: [30, 10, 70, 50]})
    assert r["identities"] == 1                              # one stable track identity across the clip
    assert len(r["boxes"]) == 6
    assert r["healed_gaps"] == 3                             # frames 1,2,3 interpolated between anchors 0 and 4
    sources = {b["source"] for b in r["boxes"]}
    assert "interpolated" in sources and "keyframe" in sources


def test_quality_flags_bad_geometry():
    tiny = annotation_quality({"bbox": [10, 10, 12, 12], "conf": 0.9, "source": "auto"})
    assert "tiny_box" in tiny["flags"] and tiny["quality"] < 0.9
    off = annotation_quality({"bbox": [10, 10, 5000, 50], "conf": 0.9, "source": "auto"})
    assert "off_screen" in off["flags"]
    good = annotation_quality({"class_id": 6, "bbox": [100, 100, 300, 300], "conf": 0.9, "source": "human"})
    assert good["quality"] > 0.8 and good["flags"] == []


def test_inter_annotator_agreement():
    same = [{"class_id": 6, "bbox": BOX}, {"class_id": 6, "bbox": [11, 11, 51, 51]}]
    assert inter_annotator_agreement(same) == 1.0
    diff = [{"class_id": 6, "bbox": BOX}, {"class_id": 11, "bbox": [500, 500, 540, 540]}]
    assert inter_annotator_agreement(diff) == 0.0


def test_gold_audit_catches_bad_annotation():
    predicted = [{"object_id": "a", "class_id": 6, "bbox": BOX},               # matches gold
                 {"object_id": "b", "class_id": 6, "bbox": [500, 500, 540, 540]}]  # seeded bad (misplaced)
    gold = [{"class_id": 6, "bbox": [11, 11, 51, 51]}]
    rep = gold_audit(predicted, gold)
    v = {x["object_id"]: x["verdict"] for x in rep["verdicts"]}
    assert v["a"] == "pass" and v["b"] == "fail"
    assert rep["n_fail"] == 1
