"""LiDAR BEV: an oriented box drawn on the bird's-eye image lifts back to the correct metric 3D cuboid,
with the height taken from the enclosed points. Pure-unit, no infra."""

from __future__ import annotations

import numpy as np

from services.lidar.bev import bev_box_to_cuboid, default_bev_params, metric_to_px, rasterize_bev


def _box_points(cx, cy, length, width, z_lo, z_hi, n=400, rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    x = rng.uniform(cx - length / 2, cx + length / 2, n)
    y = rng.uniform(cy - width / 2, cy + width / 2, n)
    z = rng.uniform(z_lo, z_hi, n)
    return np.stack([x, y, z, np.ones(n)], axis=1).astype(np.float32)


def test_bev_box_lifts_to_metric_cuboid():
    p = default_bev_params()
    # a 4 m (forward) x 2 m (lateral) object at x=20, y=0, height 1.5 m sitting on the ground
    pts = _box_points(20.0, 0.0, 4.0, 2.0, -1.7, -0.2)
    # the BEV pixel AABB of that footprint
    u_c, v_c = metric_to_px(20.0, 0.0, p)
    w_px, l_px = 2.0 / p["res"], 4.0 / p["res"]
    bbox = [u_c - w_px / 2, v_c - l_px / 2, u_c + w_px / 2, v_c + l_px / 2]

    cub = bev_box_to_cuboid(bbox, 0.0, pts, p)
    assert abs(cub["center"][0] - 20.0) < 0.2   # x forward
    assert abs(cub["center"][1] - 0.0) < 0.2    # y lateral
    assert abs(cub["size"][0] - 2.0) < 0.2      # width
    assert abs(cub["size"][1] - 4.0) < 0.2      # length
    assert abs(cub["size"][2] - 1.5) < 0.3      # height from the enclosed points
    assert cub["n_points"] > 100                # the footprint really enclosed the object


def test_bev_box_without_points_uses_default_height():
    p = default_bev_params()
    empty = np.zeros((0, 4), dtype=np.float32)
    u_c, v_c = metric_to_px(30.0, -5.0, p)
    bbox = [u_c - 10, v_c - 20, u_c + 10, v_c + 20]
    cub = bev_box_to_cuboid(bbox, 0.0, empty, p, default_h=1.6)
    assert cub["n_points"] == 0 and abs(cub["size"][2] - 1.6) < 1e-6


def test_rasterize_bev_shape_and_content():
    p = default_bev_params()
    pts = _box_points(15.0, 2.0, 4.0, 2.0, -1.7, -0.2, n=2000)
    img = rasterize_bev(pts, p)
    assert img.shape == (p["height"], p["width"], 3)
    assert int((img.sum(axis=2) > 0).sum()) > 50  # the object footprint is drawn
