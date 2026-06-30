"""M-4D.3: the optical-flow box-propagation core behind SAM-propagated tracking. A textured patch moved by
a known shift must carry the box by that shift; a multi-frame run must accumulate; a featureless target
must stop (return None) rather than drift silently."""

from __future__ import annotations

import cv2
import numpy as np

from services.temporal.sam_propagate import flow_box, flow_track

_BG = 30


def _frame(shift=(0, 0), w=240, h=240, patch=(60, 60, 140, 140)):
    rng = np.random.default_rng(0)
    canvas = np.full((h, w), _BG, np.uint8)
    px1, py1, px2, py2 = patch
    canvas[py1:py2, px1:px2] = rng.integers(0, 255, (py2 - py1, px2 - px1), dtype=np.uint8)
    if shift != (0, 0):
        mt = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        canvas = cv2.warpAffine(canvas, mt, (w, h), borderValue=_BG)
    return canvas


def test_flow_box_tracks_a_translation():
    g0, g1 = _frame(), _frame(shift=(12, 7))
    box0 = [60.0, 60.0, 140.0, 140.0]
    nb = flow_box(g0, g1, box0)
    assert nb is not None
    assert abs((nb[0] - box0[0]) - 12) < 4 and abs((nb[1] - box0[1]) - 7) < 4


def test_flow_track_accumulates_over_frames():
    frames = [_frame(shift=(i * 8, 0)) for i in range(4)]   # patch moves +8 px / frame
    boxes = flow_track(frames, [60.0, 60.0, 140.0, 140.0])
    assert len(boxes) == 3 and all(b is not None for b in boxes)
    assert boxes[-1][0] > 60 + 12                            # cumulative rightward drift


def test_flow_track_stops_on_a_featureless_frame():
    g0 = _frame()
    blank = np.full((240, 240), _BG, np.uint8)
    boxes = flow_track([g0, blank], [60.0, 60.0, 140.0, 140.0])
    assert boxes[0] is None
