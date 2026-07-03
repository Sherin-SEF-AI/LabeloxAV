"""Calibration harness (M-E calibration leg): the deterministic matching, fit/validate, and config-activation
logic. The detector-inference and VLM-judge collectors need a GPU and a VLM, so they are exercised operationally,
not here; this pins the pure logic that decides whether a curve is trustworthy and how it is deployed."""

from __future__ import annotations

import numpy as np

from services.autolabel.gold_calibrate import _ece, _fit_report, _match_frame


def test_match_frame_greedy_iou_class():
    truth = [{"class_id": 5, "bbox": [0, 0, 10, 10]}, {"class_id": 5, "bbox": [100, 100, 110, 110]}]
    dets = [
        {"class_id": 5, "bbox": [0, 0, 9, 9], "conf": 0.9},        # matches truth[0] -> correct
        {"class_id": 5, "bbox": [1, 1, 11, 11], "conf": 0.8},      # truth[0] already used -> FP
        {"class_id": 7, "bbox": [100, 100, 110, 110], "conf": 0.7},  # right box wrong class -> FP
        {"class_id": 5, "bbox": [200, 200, 210, 210], "conf": 0.6},  # no overlap -> FP
    ]
    pairs = dict(_match_frame(dets, truth, iou_thr=0.5))
    assert pairs[0.9] == 1          # the one true positive
    assert pairs[0.8] == 0 and pairs[0.7] == 0 and pairs[0.6] == 0  # duplicate, wrong-class, and miss are FPs


def test_ece_zero_on_perfect_and_positive_on_overconfident():
    ys = np.array([1.0, 0.0, 1.0, 0.0])
    assert _ece(ys.copy(), ys) < 1e-9                      # predictions equal outcomes -> ECE 0
    over = np.array([0.99, 0.99, 0.99, 0.99])
    assert _ece(over, ys) > 0.4                            # all-0.99 on 50% accuracy -> large ECE


def test_fit_report_calibrates_and_flags_trustworthy():
    # a real signal: P(correct) rises with confidence, but the raw score is overconfident (correct only ~half
    # the time at raw 0.9). A good curve pulls that down and lowers held-out ECE.
    rng = np.random.default_rng(0)
    per_frame = {}
    for f in range(40):
        pairs = []
        for _ in range(8):
            c = float(rng.uniform(0.3, 0.99))
            p_true = 0.15 + 0.5 * (c - 0.3) / 0.69          # 0.15 at 0.3 up to ~0.65 at 0.99
            pairs.append((c, int(rng.random() < p_true)))
        per_frame[f"f{f}"] = pairs
    rep = _fit_report({"source": "unit", "weights": None, "per_frame": per_frame},
                      tag="unit", storage_key="calibration/unit/test.json", val_frac=0.3, seed=1)
    assert rep["val_positives"] >= 3 and rep["val_negatives"] >= 3
    assert rep["calibrated_val_ece"] <= rep["raw_val_ece"]   # calibration does not worsen held-out ECE
    assert rep["non_degenerate"] and rep["trustworthy"]
    # the deployed curve is monotone non-decreasing and pulls the overconfident raw scores down
    xs = sorted(rep["curve"])
    ys = [rep["curve"][x] for x in xs]
    assert all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1))
    assert rep["curve"][0.9] < 0.9
