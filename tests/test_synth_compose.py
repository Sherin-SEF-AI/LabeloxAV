"""The copy-paste compose step on generated images: pure, deterministic, and refuses what it cannot place.

Every image here is drawn by the test so each expected property is checkable by hand: the background is a
flat gradient, the drivable surface is a band of rows, and the donor is a solid disc with a known mask.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from services.synth.copy_paste import (
    MAX_COVER_FRAC,
    Composite,
    Refusal,
    compose,
    cover_fraction,
    horizon_row,
    polygons_of,
    rasterize_polygons,
    reinhard_transfer,
)

W, H = 640, 480


def _background() -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[..., 0] = np.linspace(40, 200, W, dtype=np.uint8)[None, :]
    img[..., 1] = np.linspace(60, 160, H, dtype=np.uint8)[:, None]
    img[..., 2] = 90
    return img


def _drivable(rows: tuple[int, int]) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    m[rows[0]:rows[1], 100:540] = 1
    return m


def _donor(radius: int = 40) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """A donor frame with one bright disc; the mask is the disc, the box is its bounding square."""
    img = np.full((300, 300, 3), 30, dtype=np.uint8)
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img, (150, 150), radius, (230, 230, 240), -1)
    cv2.circle(mask, (150, 150), radius, 1, -1)
    return img, mask, [150.0 - radius, 150.0 - radius, 150.0 + radius, 150.0 + radius]


class TestHelpers:
    def test_polygon_round_trip(self):
        _, mask, _ = _donor()
        polys = polygons_of(mask)
        back = rasterize_polygons(polys, 300, 300)
        # A disc approximated at 1 px tolerance keeps at least 95% of its area.
        inter = np.logical_and(back, mask).sum()
        assert inter / mask.sum() > 0.95

    def test_horizon_row(self):
        assert horizon_row(240.0, 500.0, 0.0) == 240.0
        # Pitched down 5 degrees: the horizon rises above the principal point.
        assert horizon_row(240.0, 500.0, 5.0) < 240.0

    def test_cover_fraction(self):
        assert cover_fraction([0, 0, 10, 10], [5, 5, 15, 15]) == 0.25
        assert cover_fraction([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
        assert cover_fraction([0, 0, 40, 40], [5, 5, 15, 15]) == 1.0

    def test_reinhard_moves_toward_reference_without_erasing_the_donor(self):
        img, mask, _ = _donor()
        ref = np.full((50, 50, 3), (20, 120, 200), dtype=np.uint8)
        out = reinhard_transfer(img, mask, ref, 0.5)
        assert out.shape == img.shape
        before = img[mask > 0].mean(axis=0)
        after = out[mask > 0].mean(axis=0)
        # The disc moved toward the reference colour but is still the bright object it was.
        assert np.abs(after - ref.reshape(-1, 3).mean(axis=0)).sum() < np.abs(before - ref.reshape(-1, 3).mean(axis=0)).sum()
        assert after.mean() > 120
        assert reinhard_transfer(img, mask, ref, 0.0) is img


class TestCompose:
    def test_places_on_the_drivable_band_and_touches_nothing_else(self):
        bg = _background()
        img, mask, box = _donor()
        res = compose(bg, {}, _drivable((300, 460)), img, mask, box,
                      donor_horizon=150.0, target_horizon=200.0, rng=random.Random(1))
        assert isinstance(res, Composite), res
        x0, y0, x1, y1 = res.bbox
        assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
        # The paste stands on a drivable row: its bottom sits inside the band.
        assert 300 <= y1 <= 460
        assert res.polygons and all(len(p) >= 6 for p in res.polygons)
        # Outside the paste box, plus the feather margin, the background is untouched.
        touched = np.any(res.image != bg, axis=2)
        ys, xs = np.nonzero(touched)
        assert xs.min() >= x0 - 4 and xs.max() < x1 + 4 and ys.min() >= y0 - 4 and ys.max() < y1 + 4
        assert touched.any()

    def test_scale_follows_the_row_depth_model(self):
        """Same horizon offset in both frames and a one-row drivable band: the paste is the donor's size."""
        bg = _background()
        img, mask, box = _donor(radius=40)
        ground = 400
        band = np.zeros((H, W), dtype=np.uint8)
        band[ground, 100:540] = 1
        # Donor stood 190 rows below its horizon; the target row is 190 below the target horizon.
        res = compose(bg, {}, band, img, mask, box, donor_horizon=box[3] - 190.0,
                      target_horizon=float(ground - 190), rng=random.Random(3))
        assert isinstance(res, Composite), res
        assert res.scale == 1.0
        assert abs((res.bbox[3] - res.bbox[1]) - 80) <= 2 and abs((res.bbox[2] - res.bbox[0]) - 80) <= 2
        # Twice as far below the horizon: twice the size.
        res2 = compose(bg, {}, band, img, mask, box, donor_horizon=box[3] - 95.0,
                       target_horizon=float(ground - 190), rng=random.Random(3))
        assert isinstance(res2, Composite), res2
        assert res2.scale == 2.0

    def test_deterministic_under_the_same_rng(self):
        bg = _background()
        img, mask, box = _donor()
        a = compose(bg, {}, _drivable((300, 460)), img, mask, box, donor_horizon=150.0, target_horizon=200.0,
                    rng=random.Random(7))
        b = compose(bg, {}, _drivable((300, 460)), img, mask, box, donor_horizon=150.0, target_horizon=200.0,
                    rng=random.Random(7))
        assert isinstance(a, Composite) and isinstance(b, Composite)
        assert a.bbox == b.bbox and a.flipped == b.flipped and np.array_equal(a.image, b.image)

    def test_refuses_to_cover_a_background_box(self):
        bg = _background()
        img, mask, box = _donor()
        # One placement only, and a small background box sitting right under it, so the paste would hide
        # all of it.
        band = np.zeros((H, W), dtype=np.uint8)
        band[400, 320] = 1
        boxes = {"bg-1": [300.0, 340.0, 340.0, 380.0]}
        res = compose(bg, boxes, band, img, mask, box, donor_horizon=box[3] - 190.0,
                      target_horizon=float(400 - 190), rng=random.Random(1))
        assert isinstance(res, Refusal)
        assert "cover" in res.reason and f"{MAX_COVER_FRAC:.0%}" in res.reason

    def test_records_partial_cover_it_allows(self):
        bg = _background()
        img, mask, box = _donor()
        band = np.zeros((H, W), dtype=np.uint8)
        band[400, 320] = 1   # exactly one placement: centred at column 320, standing on row 400
        boxes = {"bg-1": [300.0, 380.0, 600.0, 480.0]}   # a 300x100 box the paste clips a corner of
        res = compose(bg, boxes, band, img, mask, box, donor_horizon=box[3] - 190.0,
                      target_horizon=float(400 - 190), rng=random.Random(1))
        assert isinstance(res, Composite), res
        assert 0 < res.covered["bg-1"] <= MAX_COVER_FRAC

    def test_refuses_a_thin_mask_and_no_surface(self):
        bg = _background()
        img, mask, box = _donor()
        thin = np.zeros_like(mask)
        thin[150, 110:190] = 1
        r1 = compose(bg, {}, _drivable((300, 460)), img, thin, box, donor_horizon=150.0, target_horizon=200.0,
                     rng=random.Random(1))
        assert isinstance(r1, Refusal) and "20%" in r1.reason
        r2 = compose(bg, {}, np.zeros((H, W), dtype=np.uint8), img, mask, box, donor_horizon=150.0,
                     target_horizon=200.0, rng=random.Random(1))
        assert isinstance(r2, Refusal) and "drivable" in r2.reason
        r3 = compose(bg, {}, _drivable((300, 460)), img, mask, box, donor_horizon=250.0, target_horizon=200.0,
                     rng=random.Random(1))
        assert isinstance(r3, Refusal) and "horizon" in r3.reason


class TestHoodSubtraction:
    def test_hood_rows_are_cut_from_the_surface(self, monkeypatch):
        from services.autolabel import ego_mask as em
        from services.synth import copy_paste as cp

        # A 48x64 hood grid whose bottom quarter is the bonnet, the way the estimator writes it.
        grid = tuple(tuple(1 if gy >= 36 else 0 for _ in range(64)) for gy in range(48))
        monkeypatch.setattr(em, "get_ego_mask", lambda v, c: em.EgoMask(grid=grid, area_frac=0.25))
        drivable = np.ones((480, 640), dtype=np.uint8)
        out, known = cp._without_hood(drivable, "V", "front")
        assert known
        assert out[:360].all() and not out[360:].any()
        assert drivable.all(), "the caller's mask is not modified in place"

    def test_unknown_camera_keeps_the_raw_surface(self, monkeypatch):
        from services.autolabel import ego_mask as em
        from services.synth import copy_paste as cp

        monkeypatch.setattr(em, "get_ego_mask", lambda v, c: None)
        drivable = np.ones((480, 640), dtype=np.uint8)
        out, known = cp._without_hood(drivable, "V", "front")
        assert not known and out is drivable
