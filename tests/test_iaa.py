"""Milestone I: inter-annotator agreement. Identical label sets agree perfectly; a missed box lowers
detection agreement; a class disagreement on a matched pair lowers class agreement and kappa."""

from __future__ import annotations

from services.quality.iaa import cohen_kappa, iaa_score, iou, match_boxes


def test_iou_identical_and_disjoint():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_match_is_one_to_one():
    a = [[0, 0, 10, 10], [100, 100, 110, 110]]
    b = [[0, 0, 10, 10], [100, 100, 110, 110]]
    assert len(match_boxes(a, b, 0.5)) == 2


def test_identical_sets_agree_perfectly():
    s = [{"bbox": [0, 0, 10, 10], "class_name": "car"}, {"bbox": [50, 50, 60, 60], "class_name": "truck"}]
    r = iaa_score(s, s)
    assert r["detection_agreement"] == 1.0 and r["class_agreement"] == 1.0 and r["cohen_kappa"] == 1.0


def test_missed_box_lowers_detection_agreement():
    a = [{"bbox": [0, 0, 10, 10], "class_name": "car"}, {"bbox": [50, 50, 60, 60], "class_name": "car"}]
    b = [{"bbox": [0, 0, 10, 10], "class_name": "car"}]                 # annotator B missed one
    r = iaa_score(a, b)
    assert r["n_matched"] == 1 and r["detection_agreement"] == 0.5      # 1 matched / union of 2


def test_class_disagreement_lowers_kappa():
    a = [{"bbox": [0, 0, 10, 10], "class_name": "car"}, {"bbox": [50, 50, 60, 60], "class_name": "truck"}]
    b = [{"bbox": [0, 0, 10, 10], "class_name": "car"}, {"bbox": [50, 50, 60, 60], "class_name": "bus"}]
    r = iaa_score(a, b)
    assert r["detection_agreement"] == 1.0 and r["class_agreement"] == 0.5 and r["cohen_kappa"] < 1.0


def test_kappa_perfect_on_single_shared_class():
    assert cohen_kappa(["car", "car"], ["car", "car"]) == 1.0
