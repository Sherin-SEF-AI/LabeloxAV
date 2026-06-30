"""M-4D.1: the map->frame transform behind one-shot aggregate labeling. A scan's pose maps its ego points
INTO the map, so a map-frame cuboid maps back into the scan's ego frame by the inverse pose, with the yaw
subtracting the pose's own yaw."""

from __future__ import annotations

import math

from services.lidar.aggregate.label_propagate import map_cuboid_to_frame

_I = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _rz(theta, tx=0.0, ty=0.0):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0, tx], [s, c, 0, ty], [0, 0, 1, 0], [0, 0, 0, 1]]


def test_identity_pose_is_a_passthrough():
    c, y = map_cuboid_to_frame([3.0, 1.0, 0.5], 0.4, _I)
    assert c == [3.0, 1.0, 0.5] and abs(y - 0.4) < 1e-6


def test_pure_translation_inverts():
    # pose maps frame->map by +10 x, so map (10,0,0) is the frame origin
    pose = [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    c, _ = map_cuboid_to_frame([10.0, 0.0, 0.0], 0.0, pose)
    assert abs(c[0]) < 1e-6 and abs(c[1]) < 1e-6 and abs(c[2]) < 1e-6


def test_yaw_subtracts_pose_yaw():
    pose = _rz(math.radians(90))
    _, y = map_cuboid_to_frame([0.0, 0.0, 0.0], math.radians(30), pose)
    assert abs(y - math.radians(30 - 90)) < 1e-4   # the function rounds yaw to 5 places


def test_rotation_and_translation_round_trip():
    # frame point (1,0,0) under Rz(90)+t(5,0) maps to map (5,1,0); so map (5,1,0) -> frame (1,0,0)
    pose = _rz(math.radians(90), tx=5.0)
    c, _ = map_cuboid_to_frame([5.0, 1.0, 0.0], 0.0, pose)
    assert abs(c[0] - 1.0) < 1e-4 and abs(c[1]) < 1e-4
