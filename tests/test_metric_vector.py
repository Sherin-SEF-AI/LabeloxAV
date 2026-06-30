"""Milestone F: 3D metric polylines. Metric length in the ego frame, and georeferencing an ego polyline to
world (forward is north, the ego right side is east, at heading north)."""

from __future__ import annotations

from services.hdmap.metric_vector import ego_polyline_to_world, polyline_length_m


def test_polyline_length_metres():
    assert abs(polyline_length_m([[0, 0], [3, 0], [3, 4]]) - 7.0) < 1e-9   # 3 + 4
    assert polyline_length_m([[1, 1]]) == 0.0


def test_forward_point_is_north():
    (lon, lat), = ego_polyline_to_world([[10.0, 0.0]], lat=0.0, lon=0.0, heading_rad=0.0)
    assert lat > 0 and abs(lon) < 1e-9                  # 10 m forward at heading north -> +lat


def test_ego_right_point_is_east():
    # ego y is left, so y = -5 is 5 m to the right; at heading north that is east (+lon)
    (lon, lat), = ego_polyline_to_world([[0.0, -5.0]], lat=0.0, lon=0.0, heading_rad=0.0)
    assert lon > 0 and abs(lat) < 1e-6
