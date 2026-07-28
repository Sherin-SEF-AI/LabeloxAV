"""The evaluation metrics the system could not compute.

The audit found that evaluation was 2D-box only. The system labels masks, cuboids, tracks, and lanes, and
could score none of them: `ModelRegistry.gold_metrics` even documents itself as carrying "mask_iou, MOTA,
IDF1", which nothing produced. `core/accel/mask_iou.py` existed and no evaluation path called it.

These are pure numeric kernels, so the tests are exact rather than approximate: each asserts a value that can
be derived by hand, and each metric is shown to react to the specific failure it exists to catch."""
from __future__ import annotations

import numpy as np
import pytest

from core.accel.ap3d import bev_iou, cuboid_average_precision, evaluate_cuboids, match_cuboids
from core.accel.lane_metrics import lane_distance, lane_f1, sample_lane_at_rows
from core.accel.mask_ap import boundary_f1, mask_ap_50_95, mask_iou, match_masks, polygons_to_mask
from core.accel.tracking_metrics import Detection, evaluate_tracking


def _square(x0: int, y0: int, side: int, size: int = 100) -> np.ndarray:
    m = np.zeros((size, size), dtype=bool)
    m[y0:y0 + side, x0:x0 + side] = True
    return m


# ---------------- segmentation ----------------

def test_mask_iou_is_exact_for_known_overlap():
    a = _square(0, 0, 10)      # 100 px
    b = _square(5, 0, 10)      # 100 px, overlapping 50
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, b) == pytest.approx(50 / 150)


def test_disjoint_masks_score_zero():
    assert mask_iou(_square(0, 0, 10), _square(50, 50, 10)) == 0.0


def test_polygons_rasterize_to_the_expected_area():
    # cv2.fillPoly includes both boundary rows and columns, so a 10..30 square covers 21x21 pixels, not 20x20.
    m = polygons_to_mask([[10, 10, 30, 10, 30, 30, 10, 30]], 100, 100)
    assert m.sum() == 21 * 21


def test_mask_ap_is_perfect_when_the_mask_is_exact_and_zero_when_it_is_not_close():
    gt = _square(10, 10, 40)
    assert mask_ap_50_95([gt], [0.9], [1], [gt], [1])["ap50"] == 1.0
    far = _square(60, 60, 40)      # no overlap at all
    assert mask_ap_50_95([far], [0.9], [1], [gt], [1])["ap50"] == 0.0


def test_mask_ap_separates_a_loose_mask_from_a_tight_one():
    # The whole reason to sweep IoU thresholds: a loose mask clears 0.5 but fails the strict end, so AP@50
    # alone would call a blob and a traced silhouette equally good.
    gt = _square(20, 20, 40)
    loose = _square(15, 15, 50)
    r = mask_ap_50_95([loose], [0.9], [1], [gt], [1])
    assert r["ap50"] == 1.0
    assert r["ap50_95"] < r["ap50"]


def test_mask_matching_is_class_aware():
    m = _square(10, 10, 20)
    # right shape, wrong class: not a true positive
    r = match_masks([m], [0.9], [2], [m], [1])
    assert r["n_tp"] == 0 and r["n_fp"] == 1 and r["n_fn"] == 1


def test_mask_ap_is_none_without_ground_truth():
    # AP is undefined with no ground truth. Zero would read as failure and one as success; both are lies.
    assert mask_ap_50_95([_square(0, 0, 5)], [0.9], [1], [], [])["ap50"] is None


def test_boundary_f1_catches_a_mask_that_ious_well_but_traces_badly():
    gt = _square(20, 20, 60)
    # a mask inset on every side: high IoU (interior dominates), visibly wrong silhouette
    inset = _square(24, 24, 52)
    assert mask_iou(inset, gt) > 0.7
    assert boundary_f1(inset, gt) < 0.5


# ---------------- 3D cuboids ----------------

def _box(cx=0.0, cy=0.0, cz=0.0, w=4.0, length=2.0, h=2.0, yaw=0.0) -> dict:
    return {"center": [cx, cy, cz], "dims": [w, length, h], "yaw": yaw}


def test_bev_ignores_height_while_3d_does_not():
    # The reason both are reported. Identical footprint, double the height: the footprints agree exactly
    # (BEV 1.0) while the volumes overlap only half (3D 0.5). At a strict threshold the box passes the
    # planning-relevant question and fails the volumetric one, which a single number could not express.
    a, tall = _box(h=2.0), _box(h=4.0)
    assert bev_iou(a, tall) == pytest.approx(1.0)
    e = evaluate_cuboids([tall], [0.9], [1], [a], [1], iou_thr=0.7)
    assert e["ap_bev"] == 1.0
    assert e["ap_3d"] == 0.0


def test_perfect_cuboid_scores_one_with_no_localisation_error():
    b = _box()
    e = evaluate_cuboids([b], [0.9], [1], [b], [1])
    assert e["ap_3d"] == 1.0 and e["ap_bev"] == 1.0
    assert e["translation_error_m"] == 0.0
    assert e["orientation_error_rad"] == 0.0


def test_translation_error_reports_the_offset_ap_hides():
    # A box shifted 0.5m still matches at a loose threshold, so AP stays perfect while the position is wrong.
    gt, shifted = _box(), _box(cx=0.5)
    e = evaluate_cuboids([shifted], [0.9], [1], [gt], [1], iou_thr=0.3)
    assert e["ap_3d"] == 1.0
    assert e["translation_error_m"] == pytest.approx(0.5, abs=1e-6)


def test_orientation_error_wraps_so_a_symmetric_flip_is_not_a_large_error():
    gt = _box(yaw=0.0)
    flipped = _box(yaw=float(np.pi * 2))    # same orientation, expressed differently
    e = evaluate_cuboids([flipped], [0.9], [1], [gt], [1], iou_thr=0.3)
    assert e["orientation_error_rad"] == pytest.approx(0.0, abs=1e-6)


def test_cuboid_matching_is_class_aware():
    b = _box()
    m = match_cuboids([b], [0.9], [2], [b], [1])
    assert m["n_tp"] == 0 and m["n_fn"] == 1


def test_cuboid_ap_is_none_without_ground_truth():
    assert cuboid_average_precision([_box()], [0.9], [1], [], []) is None
    assert evaluate_cuboids([_box()], [0.9], [1], [], [])["measured"] is False


# ---------------- tracking ----------------

def _seq(track_id: str, n: int, box=(0.0, 0.0, 10.0, 10.0)) -> list[Detection]:
    return [Detection(f, track_id, box) for f in range(n)]


def test_perfect_tracking_scores_one_across_all_three_metrics():
    gt = _seq("g1", 5)
    r = evaluate_tracking(_seq("p1", 5), gt)
    assert r["mota"] == 1.0 and r["idf1"] == 1.0 and r["hota"] == 1.0
    assert r["id_switches"] == 0


def test_an_identity_switch_costs_idf1_more_than_mota():
    # This is why both are reported. Detection is flawless; only the identity breaks. MOTA charges one
    # switch out of five detections; IDF1 charges the whole broken half of the track.
    gt = _seq("g1", 6)
    swapped = [Detection(f, "p1" if f < 3 else "p2", (0.0, 0.0, 10.0, 10.0)) for f in range(6)]
    r = evaluate_tracking(swapped, gt)
    assert r["id_switches"] == 1
    assert r["idf1"] < r["mota"]


def test_misses_and_false_positives_are_counted_separately():
    gt = _seq("g1", 4)
    r = evaluate_tracking(_seq("p1", 2), gt)          # tracker stops halfway
    assert r["misses"] == 2 and r["false_positives"] == 0
    assert r["mota"] == pytest.approx(0.5)

    r2 = evaluate_tracking(_seq("p1", 4) + [Detection(0, "p2", (50.0, 50.0, 60.0, 60.0))], gt)
    assert r2["false_positives"] == 1 and r2["misses"] == 0


def test_mota_can_go_negative_and_is_not_clamped():
    # A tracker emitting far more false positives than there are objects is worse than emitting nothing;
    # clamping at zero would hide exactly that.
    gt = _seq("g1", 2)
    noise = [Detection(f, f"n{i}", (100.0 + 20 * i, 100.0, 110.0 + 20 * i, 110.0))
             for f in range(2) for i in range(4)]
    assert evaluate_tracking(_seq("p1", 2) + noise, gt)["mota"] < 0


def test_empty_input_is_unmeasured_not_a_perfect_score():
    assert evaluate_tracking([], [])["measured"] is False


# ---------------- lanes ----------------

def _line(x: float, height: int = 100) -> list[list[float]]:
    return [[x, 0.0], [x, float(height)]]


def test_identical_lanes_match_with_zero_error():
    r = lane_f1([_line(50)], [_line(50)], width=200, height=100)
    assert r["f1"] == 1.0 and r["mean_lateral_error_px"] == pytest.approx(0.0)


def test_a_lane_outside_the_tolerance_is_a_miss_and_a_false_positive():
    # 100px apart on a 200px-wide image, far beyond the ~3.7px tolerance.
    r = lane_f1([_line(150)], [_line(50)], width=200, height=100)
    assert r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 1 and r["f1"] == 0.0


def test_lane_distance_is_the_mean_lateral_offset():
    assert lane_distance(_line(55), _line(50), height=100) == pytest.approx(5.0)


def test_lanes_that_never_overlap_vertically_do_not_match():
    # Without the vertical-extent check these would compare on interpolated values they never covered.
    top = [[50.0, 0.0], [50.0, 10.0]]
    bottom = [[50.0, 90.0], [50.0, 100.0]]
    assert lane_distance(top, bottom, height=100) == float("inf")


def test_sampling_returns_nan_outside_a_lanes_own_extent():
    rows = np.array([0.0, 50.0, 99.0])
    xs = sample_lane_at_rows([[10.0, 40.0], [10.0, 60.0]], rows)
    assert np.isnan(xs[0]) and np.isnan(xs[2])
    assert xs[1] == pytest.approx(10.0)


def test_lane_f1_is_unmeasured_without_ground_truth():
    assert lane_f1([_line(50)], [], width=200, height=100)["measured"] is False
