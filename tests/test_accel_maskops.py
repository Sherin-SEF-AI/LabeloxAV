"""Mask-ops (SAM postprocess) verification: morphology matches cv2 on the interior, the area filter drops small
fragments, the RLE codec round-trips exactly, and boundary tightness ranks a compact mask above a ragged one."""

import numpy as np
import pytest

from core.accel.maskops import (
    boundary_tightness,
    dilate,
    erode,
    morph_open,
    remove_small_components,
    rle_decode,
    rle_encode,
)


def test_morphology_matches_cv2_interior():
    # This exercises the torch kernel itself (there is no numpy fallback for it), so a box without torch
    # skips rather than failing. Deliberately importorskip and NOT the `gpu` marker: the kernel runs on CPU
    # torch, and the marker would deselect it from `make test-unit` on every box that could run it.
    pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    mask = (rng.random((80, 100)) > 0.5).astype(np.float32)
    k = 3
    kernel = np.ones((k, k), np.uint8)
    for op, ref in [(dilate, cv2.dilate), (erode, cv2.erode)]:
        got = op(mask, k, device="cpu")
        cref = ref(mask, kernel).astype(np.float32)
        # compare interior (border convention differs between replicate-pad pooling and cv2)
        assert np.array_equal(got[k:-k, k:-k] > 0.5, cref[k:-k, k:-k] > 0.5)


def test_open_removes_speckle():
    # This exercises the torch kernel itself (there is no numpy fallback for it), so a box without torch
    # skips rather than failing. Deliberately importorskip and NOT the `gpu` marker: the kernel runs on CPU
    # torch, and the marker would deselect it from `make test-unit` on every box that could run it.
    pytest.importorskip("torch")
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[10:30, 10:30] = 1.0                               # a solid block
    mask[2, 2] = 1.0                                       # an isolated speckle
    opened = morph_open(mask, 3, device="cpu") > 0.5
    assert not opened[2, 2]                                # speckle gone
    assert opened[20, 20]                                  # block survives


def test_area_filter_drops_fragments():
    mask = np.zeros((50, 50), dtype=bool)
    mask[5:25, 5:25] = True                                # 400-px component
    mask[40, 40] = True                                    # 1-px fragment
    mask[45, 45] = True
    cleaned = remove_small_components(mask, min_area=10)
    assert cleaned[10, 10] and not cleaned[40, 40] and not cleaned[45, 45]


def test_rle_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(5):
        m = rng.random((30, 40)) > 0.6
        assert np.array_equal(rle_decode(rle_encode(m)), m)
    # all-zero and all-one edge cases
    assert np.array_equal(rle_decode(rle_encode(np.zeros((8, 8), bool))), np.zeros((8, 8), bool))
    assert np.array_equal(rle_decode(rle_encode(np.ones((8, 8), bool))), np.ones((8, 8), bool))


def test_boundary_tightness_ranks_compact_higher():
    # This exercises the torch kernel itself (there is no numpy fallback for it), so a box without torch
    # skips rather than failing. Deliberately importorskip and NOT the `gpu` marker: the kernel runs on CPU
    # torch, and the marker would deselect it from `make test-unit` on every box that could run it.
    pytest.importorskip("torch")
    compact = np.zeros((40, 40), dtype=np.float32)
    compact[10:30, 10:30] = 1.0                            # solid square: mostly interior
    ragged = np.zeros((40, 40), dtype=np.float32)
    ragged[::2, ::2] = 1.0                                 # a sparse grid: all boundary
    tc = boundary_tightness(compact, device="cpu")
    tr = boundary_tightness(ragged, device="cpu")
    assert tc > tr and tc > 0.3
