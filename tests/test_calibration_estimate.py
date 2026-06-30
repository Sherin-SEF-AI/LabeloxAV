"""M-CAL.3b: monocular estimation. The vanishing point of converging road lines gives the horizon, the
horizon gives the camera pitch, and EXIF (when present) gives the focal. These drive the pure geometry and
a synthetic road image so the estimate is deterministic."""

from __future__ import annotations

import math

import cv2
import numpy as np

from services.calibration.estimate import (
    estimate_frame,
    focal_from_exif,
    pitch_from_vp,
    vanishing_point,
)


def test_focal_from_35mm_equivalent():
    assert abs(focal_from_exif({"FocalLengthIn35mmFilm": 27}, 1920) - (27 / 36 * 1920)) < 1e-6
    assert focal_from_exif({}, 1920) is None
    assert focal_from_exif(None, 1920) is None


def test_vanishing_point_of_two_converging_lines():
    # two segments that both pass through (960, 400)
    vp = vanishing_point([(800, 720, 960, 400), (1120, 720, 960, 400)])
    assert vp is not None
    assert abs(vp[0] - 960) < 1.0 and abs(vp[1] - 400) < 1.0


def test_vanishing_point_needs_two_lines():
    assert vanishing_point([(0, 0, 1, 1)]) is None


def test_pitch_from_vp_sign_and_scale():
    assert abs(pitch_from_vp(540.0, 2870.0, 540.0)) < 1e-9          # VP at the centre -> level
    # a VP one focal-length above the centre is a 45 degree downward pitch
    assert abs(pitch_from_vp(540.0 - 2870.0, 2870.0, 540.0) - math.radians(45)) < 1e-6


def test_estimate_frame_on_a_synthetic_road():
    # a black frame with two white lane lines converging above the centre (a downward-pitched camera)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    vp_u, vp_v = 640, 300
    cv2.line(img, (300, 700), (vp_u, vp_v), (255, 255, 255), 6)
    cv2.line(img, (980, 700), (vp_u, vp_v), (255, 255, 255), 6)
    est = estimate_frame(img, 640.0, 360.0, 1000.0)
    assert est is not None
    assert abs(est["vp"][1] - vp_v) < 30          # recovered horizon near the drawn VP
    assert est["pitch_deg"] > 0                    # VP above centre -> a real downward pitch
