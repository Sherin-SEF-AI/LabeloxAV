"""Milestone F: occupancy voxelization from geometry. A point makes its voxel occupied; the ray from the
sensor origin to it carves the intermediate voxels free; everything else is unknown. min_points filters
sparse voxels and out-of-bounds points are ignored."""

from __future__ import annotations

from services.lidar.occupancy import _bresenham3d, voxelize_occupancy

_BOUNDS = [0, -2, -2, 10, 2, 2]


def test_bresenham_straight_line():
    assert _bresenham3d((0, 0, 0), (3, 0, 0)) == [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]


def test_single_point_occupied_with_ray_carved_free():
    occ = voxelize_occupancy([[5.0, 0.0, 0.0]], origin=[0.0, 0.0, 0.0], bounds=_BOUNDS, voxel_size=1.0)
    assert occ["occupied"] == 1
    assert occ["free"] == 5                      # 5 voxels along the ray before the occupied one
    total = occ["dims"][0] * occ["dims"][1] * occ["dims"][2]
    assert occ["unknown"] == total - 6


def test_min_points_filters_sparse_voxels():
    occ = voxelize_occupancy([[5.0, 0.0, 0.0], [5.1, 0.0, 0.0], [2.0, 0.0, 0.0]],
                             origin=[0, 0, 0], bounds=_BOUNDS, voxel_size=1.0, min_points=2)
    assert occ["occupied"] == 1                  # only the voxel with two points clears min_points


def test_out_of_bounds_points_ignored():
    occ = voxelize_occupancy([[100.0, 0.0, 0.0]], origin=[0, 0, 0], bounds=_BOUNDS, voxel_size=1.0)
    assert occ["occupied"] == 0
