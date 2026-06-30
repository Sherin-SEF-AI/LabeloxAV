"""Milestone F: BEV label surface. Oriented-footprint membership (including rotation) and rasterizing
cuboid footprints onto a top-down class grid, with painter's-order overwrite on overlap."""

from __future__ import annotations

from services.lidar.bev_labels import in_oriented_box, rasterize_label_bev

# small 10x10 m grid at 1 m/cell
_P = {"x_min": 0.0, "x_max": 10.0, "y_min": -5.0, "y_max": 5.0, "res": 1.0, "width": 10, "height": 10}


def test_oriented_box_membership():
    # box length 4 (x), width 2 (y) at origin, no rotation
    assert in_oriented_box(1.5, 0.0, 0, 0, 4, 2, 0.0) is True       # within length/2
    assert in_oriented_box(0.0, 1.5, 0, 0, 4, 2, 0.0) is False      # beyond width/2


def test_rotation_swaps_axes():
    # rotated 90 deg, the long axis now runs along y
    assert in_oriented_box(0.0, 1.8, 0, 0, 4, 2, 3.14159 / 2) is True
    assert in_oriented_box(1.8, 0.0, 0, 0, 4, 2, 3.14159 / 2) is False


def test_rasterize_one_cuboid_covers_its_footprint():
    cubs = [{"center": [5.0, 0.0, 0.0], "dims": [4.0, 2.0, 1.5], "yaw": 0.0, "class_id": 3}]
    r = rasterize_label_bev(cubs, _P)
    assert 5 <= r["labeled_cells"] <= 12        # ~4x2 = 8 m^2 at 1 m/cell, rasterization tolerance
    assert r["class_cell_counts"][3] == r["labeled_cells"]


def test_painter_order_overwrites_on_overlap():
    a = {"center": [5.0, 0.0, 0.0], "dims": [4.0, 4.0, 1.0], "yaw": 0.0, "class_id": 1}
    b = {"center": [5.0, 0.0, 0.0], "dims": [4.0, 4.0, 1.0], "yaw": 0.0, "class_id": 2}
    r = rasterize_label_bev([a, b], _P)          # b drawn last
    assert 2 in r["class_cell_counts"] and 1 not in r["class_cell_counts"]


def test_empty_input_is_empty_grid():
    r = rasterize_label_bev([], _P)
    assert r["labeled_cells"] == 0 and r["grid_shape"] == [10, 10]
