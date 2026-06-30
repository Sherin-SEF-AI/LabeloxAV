"""Ingest quality gate: a corrupted noise frame must be rejected (not scored as maximally sharp), a normal
textured frame accepted, and a flat/blurry frame still rejected."""

from __future__ import annotations

import numpy as np

from core.config import IngestSettings
from services.ingest.quality import score_frame

_CFG = IngestSettings()


def _noise():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


def _normal():
    # mid-gray background with a grid of edges -> healthy Laplacian variance, mid luma, low clipping
    img = np.full((480, 640, 3), 110, np.uint8)
    img[::20, :] = 180
    img[:, ::20] = 180
    return img


def _flat():
    return np.full((480, 640, 3), 110, np.uint8)   # no edges -> blur ~0


def test_noise_frame_rejected_not_called_sharp():
    r = score_frame(_noise(), _CFG)
    assert r.accepted is False
    assert any("noise" in reason for reason in r.reasons)
    assert r.score < 0.5                            # was ~0.90 before the fix


def test_normal_frame_accepted():
    r = score_frame(_normal(), _CFG)
    assert r.accepted is True and r.score > 0.5


def test_flat_frame_still_rejected_as_blurry():
    r = score_frame(_flat(), _CFG)
    assert r.accepted is False
    assert any("blur" in reason and "noise" not in reason for reason in r.reasons)


def test_noise_scores_below_normal():
    assert score_frame(_noise(), _CFG).score < score_frame(_normal(), _CFG).score
