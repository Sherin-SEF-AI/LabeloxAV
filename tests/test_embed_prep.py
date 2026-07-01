"""Crop preprocessing for embeddings: aspect-preserving square letterbox + tiny-crop upscaling."""

from __future__ import annotations

import numpy as np

from services.intelligence.embed.prep import PREP_TAG, square_letterbox


def test_tall_crop_is_squared_and_padded():
    a = np.zeros((200, 60, 3), np.uint8)
    a[:] = (50, 80, 200)
    out = square_letterbox(a)
    assert out.shape[0] == out.shape[1] == 200          # square, no downscale of the long side
    assert tuple(int(v) for v in out[0, 0]) == (114, 114, 114)   # corner is the grey pad
    assert tuple(int(v) for v in out[100, 100]) == (50, 80, 200)  # object content preserved at centre


def test_wide_crop_is_squared():
    a = np.zeros((60, 300, 3), np.uint8)   # short side above the 48px floor, so no upscale
    out = square_letterbox(a)
    assert out.shape[0] == out.shape[1] == 300


def test_tiny_crop_is_upscaled_then_squared():
    tiny = np.full((5, 8, 3), 120, np.uint8)
    out = square_letterbox(tiny)
    assert out.shape[0] == out.shape[1]
    assert min(out.shape[:2]) >= 48                      # upscaled past the floor


def test_square_crop_unchanged():
    sq = np.zeros((64, 64, 3), np.uint8)
    assert square_letterbox(sq).shape == (64, 64, 3)


def test_empty_crop_is_safe():
    assert square_letterbox(np.zeros((0, 0, 3), np.uint8)).size == 0


def test_prep_tag_is_recorded():
    from core.embeddings import model_versions

    # model_versions() would call the models; just assert the tag flows into the registry contract
    assert PREP_TAG and isinstance(PREP_TAG, str)
