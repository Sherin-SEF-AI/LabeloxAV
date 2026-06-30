"""9-DOF lift: the cuboid's pitch and roll come from the local contact-ground normal so a vehicle on a ramp
or banked surface tilts with it, instead of the old hard-coded 0. These prove the normal-to-tilt geometry
(in the box yaw frame), the clamp, and that fit_cuboid populates a real tilt on a synthetic ramp."""

from __future__ import annotations

import math

import numpy as np

from services.lidar.detect3d.lift import fit_cuboid, ground_tilt


def _plane_points(normal, n=240) -> np.ndarray:
    """n points sampled on the plane through the origin with the given normal."""
    nrm = np.asarray(normal, dtype=float)
    nrm = nrm / np.linalg.norm(nrm)
    a = np.array([1.0, 0.0, 0.0]) if abs(nrm[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(nrm, a)
    u /= np.linalg.norm(u)
    w = np.cross(nrm, u)
    rng = np.random.default_rng(0)
    s, t = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)
    return (s[:, None] * u + t[:, None] * w).astype(np.float32)


def test_flat_ground_has_no_tilt():
    p, r = ground_tilt(_plane_points([0, 0, 1]), 0.0, 1.0)
    assert abs(p) < 1e-5 and abs(r) < 1e-5


def test_forward_ramp_is_pitch_not_roll():
    t = 0.2
    p, r = ground_tilt(_plane_points([-math.sin(t), 0, math.cos(t)]), 0.0, 1.0)
    assert abs(p - (-t)) < 1e-3   # z = x tan t -> normal (-sin t, 0, cos t) -> pitch = -t
    assert abs(r) < 1e-3


def test_side_bank_is_roll_not_pitch():
    t = 0.25
    p, r = ground_tilt(_plane_points([0, -math.sin(t), math.cos(t)]), 0.0, 1.0)
    assert abs(r - (-t)) < 1e-3
    assert abs(p) < 1e-3


def test_yaw_rotates_a_forward_ramp_into_roll():
    t = 0.2
    # the same forward ramp, but the box faces +y (yaw 90 deg): the slope is now across the box -> roll
    p, r = ground_tilt(_plane_points([-math.sin(t), 0, math.cos(t)]), math.pi / 2, 1.0)
    assert abs(p) < 1e-3
    assert abs(abs(r) - t) < 1e-3


def test_tilt_is_clamped():
    t = 0.6
    p, _ = ground_tilt(_plane_points([-math.sin(t), 0, math.cos(t)]), 0.0, 0.35)
    assert abs(p) <= 0.35 + 1e-9


def test_sparse_ground_is_flat():
    assert ground_tilt(np.zeros((3, 3), dtype=np.float32), 0.0, 1.0) == (0.0, 0.0)


def test_fit_cuboid_populates_tilt_on_a_ramp():
    rng = np.random.default_rng(1)
    t = 0.18
    # a tilted ground patch (forward ramp rising toward +x): z = x tan t
    gx, gy = rng.uniform(8, 12, 500), rng.uniform(-2, 2, 500)
    ground = np.stack([gx, gy, gx * math.tan(t)], 1)
    # an object box (length ~4 along x, width ~2 along y) resting on the ramp
    ox, oy = rng.uniform(8, 12, 500), rng.uniform(-1, 1, 500)
    obj = np.stack([ox, oy, ox * math.tan(t) + rng.uniform(0.35, 1.6, 500)], 1)
    pts = np.concatenate([ground, obj], 0).astype(np.float32)
    plane = [-math.tan(t), 0.0, 1.0, 0.0]   # z - x tan t = 0

    cub = fit_cuboid(pts, plane, depth_gate=12.0, min_points=10)
    assert cub is not None
    assert cub["pitch"] != 0.0                 # not the old hard-coded zero
    assert abs(cub["pitch"]) > 0.05            # a real forward tilt was estimated
    assert abs(cub["roll"]) < 0.12             # a forward ramp is pitch, not roll
