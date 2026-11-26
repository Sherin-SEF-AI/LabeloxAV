"""The real SAM 2 video model follows an object it was shown once (services/temporal/sam2_video.py).

A textured disc drifts and grows across a noisy background; the tracker gets one click on the middle frame
and must find the disc on every other frame, forward and backward, without being told where it went.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

H, W, N = 360, 640, 9


def _frames():
    import cv2

    rng = np.random.default_rng(11)
    bg = rng.integers(60, 120, size=(H, W, 3), dtype=np.uint8)
    imgs, masks = [], []
    for i in range(N):
        img = bg.copy()
        cx, cy, r = 160 + 30 * i, 180, 40 + 3 * i
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (cx, cy), r, 1, -1)
        img[m.astype(bool)] = (30, 200, 240)
        cv2.circle(img, (cx - r // 3, cy - r // 3), r // 4, (20, 20, 160), -1)
        imgs.append(img)
        masks.append(m.astype(bool))
    return imgs, masks


def test_sam2_video_tracks_a_moving_object_from_one_click():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from services.temporal.sam2_video import Prompt, load_tracker, unload

    imgs, truth = _frames()
    mid = N // 2
    cx = 160 + 30 * mid
    tracker = load_tracker()
    try:
        out = {fm.index: fm for fm in tracker.track(lambda i: imgs[i], N, [Prompt(index=mid, points=[[cx, 180]],
                                                                                  labels=[1])])}
    finally:
        unload()
    assert sorted(out) == list(range(N))
    ious = []
    for i in range(N):
        assert out[i].visible, f"lost the object on frame {i}"
        inter = np.logical_and(out[i].mask, truth[i]).sum()
        union = np.logical_or(out[i].mask, truth[i]).sum()
        ious.append(inter / union)
    assert min(ious) > 0.8, ious
