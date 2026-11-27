"""The local lane proposer: paint from the segmenter, Hough only as a named fallback."""

from __future__ import annotations

import cv2
import numpy as np

from services.autolabel import drivable
from services.autolabel.lane import detect


def _broken_diagonal(h: int = 720, w: int = 1280) -> np.ndarray:
    """One straight diagonal line with a gap wider than Hough's maxLineGap, so it comes back as two pieces."""
    img = np.zeros((h, w, 3), np.uint8)
    p0, p1 = np.array([500, 340]), np.array([1100, 710])
    a, b = p0 + (p1 - p0) * 0.0, p0 + (p1 - p0) * 0.42
    c, d = p0 + (p1 - p0) * 0.58, p0 + (p1 - p0) * 1.0
    for s, e in ((a, b), (c, d)):
        cv2.line(img, tuple(int(v) for v in s), tuple(int(v) for v in e), (255, 255, 255), 6)
    return img


def test_hough_keeps_one_lane_for_two_pieces_of_one_line():
    # The old corridor test compared segment midpoints in x, so two pieces of one diagonal line, sitting at
    # different heights, came back as two lanes drawn on top of each other.
    lanes = detect._classical(_broken_diagonal(), max_lanes=6)
    assert len(lanes) == 1, lanes


def test_x_at_extrapolates_along_the_segment():
    assert detect._x_at([[0.0, 0.0], [10.0, 10.0]], 20.0) == 20.0
    assert detect._x_at([[0.0, 5.0], [10.0, 5.0]], 99.0) == 5.0   # horizontal: its own middle


def test_marking_path_is_used_and_tagged(monkeypatch):
    monkeypatch.setattr(detect, "_from_markings", lambda img, n: [[[1.0, 2.0], [3.0, 4.0]]])
    polys, tag = detect.propose_lanes_tagged(np.zeros((10, 10, 3), np.uint8))
    assert tag == detect.TAG_MARKING and polys == [[[1.0, 2.0], [3.0, 4.0]]]


def test_hough_fallback_says_so_in_its_tag(monkeypatch):
    def boom(img, n):
        raise drivable.DrivableUnavailable("no model")
    monkeypatch.setattr(detect, "_from_markings", boom)
    _polys, tag = detect.propose_lanes_tagged(_broken_diagonal())
    # Every local lane used to be tagged "clrernet:local" whatever drew it.
    assert tag == detect.TAG_HOUGH


def test_classical_setting_skips_the_segmenter(monkeypatch):
    from core.config import get_settings

    def never(img, n):
        raise AssertionError("the segmenter was asked")
    monkeypatch.setattr(detect, "_from_markings", never)
    monkeypatch.setattr(get_settings().models.lane, "model", "classical")
    assert detect.propose_lanes_tagged(_broken_diagonal())[1] == detect.TAG_HOUGH
    assert detect.model_tag() == detect.TAG_HOUGH


def test_marking_ids_take_lane_paint_but_not_crosswalks_or_kerbs():
    id2label = {"2": "Curb", "23": "Lane Marking - Crosswalk", "24": "Lane Marking - General",
                "13": "Road", "99": "Lane Marking - Stop Line"}
    assert drivable._marking_ids(id2label) == [24]
