"""ORACLYX consensus tests: fusion and the auto-label paths agreeing auto-accepts the pseudo-label; a
disagreement routes the sample to the human queue (M5 acceptance)."""

from services.oraclyx.consensus import iou, vote

BOX = [10.0, 10.0, 50.0, 50.0]


def _p(path, cls, box=None, conf=0.8):
    return {"path": path, "class_id": cls, "bbox": box or BOX, "conf": conf}


def test_iou_basic():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-6


def test_consensus_when_paths_agree():
    fusion = _p("fusion", 6, conf=0.9)
    paths = [_p("path_a", 6), _p("path_b", 6), _p("path_c", 6)]
    r = vote(fusion, paths)
    assert r["consensus"] is True and r["agree_count"] == 4
    assert r["consensus_score"] > 0.9


def test_no_consensus_on_class_disagreement_routes_to_human():
    fusion = _p("fusion", 6, conf=0.9)
    paths = [_p("path_a", 11), _p("path_b", 25), _p("path_c", 3)]  # all disagree on class
    r = vote(fusion, paths)
    assert r["consensus"] is False and r["agree_count"] == 1


def test_no_consensus_on_low_overlap():
    fusion = _p("fusion", 6)
    paths = [_p("path_a", 6, box=[200, 200, 240, 240]),  # right class, no overlap
             _p("path_b", 6, box=[201, 201, 241, 241]),
             _p("path_c", 6, box=[202, 202, 242, 242])]
    r = vote(fusion, paths)
    assert r["consensus"] is False


def test_partial_agreement_meets_min_agree():
    fusion = _p("fusion", 6)
    paths = [_p("path_a", 6), _p("path_b", 6), _p("path_c", 11)]  # 2 of 3 agree -> 3 voters total agree
    r = vote(fusion, paths, min_agree=3)
    assert r["consensus"] is True and r["agree_count"] == 3
