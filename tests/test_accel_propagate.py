"""Label-propagation kernels verification: batched SLERP matches core.geometry.slerp as a rotation, box
interpolation matches recall.interp_box, and the flow warp is correct (identity flow is a no-op, a constant
flow shifts a box/mask by the flow)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.accel.propagate import interp_boxes, slerp_batch, warp_boxes_by_flow, warp_masks_by_flow
from core.geometry import slerp as geom_slerp
from services.recall.recover import interp_box


def _quat_same_rotation(q1, q2):
    return abs(abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)))) - 1.0) < 1e-6


def test_slerp_matches_geometry_reference():
    rng = np.random.default_rng(0)
    M = 50
    q0 = Rotation.random(M, random_state=rng).as_quat()
    q1 = Rotation.random(M, random_state=rng).as_quat()
    for t in (0.0, 0.25, 0.5, 0.9, 1.0):
        got = slerp_batch(q0, q1, t)
        for i in range(M):
            assert _quat_same_rotation(got[i], geom_slerp(q0[i], q1[i], t))


def test_interp_boxes_matches_reference():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 500, size=(30, 4))
    b = rng.uniform(0, 500, size=(30, 4))
    for t in (0.0, 0.3, 1.0):
        got = interp_boxes(a, b, t)
        for i in range(30):
            assert np.allclose(got[i], interp_box(a[i], b[i], t), atol=1e-9)


def test_warp_boxes_constant_flow_shifts():
    H, W = 200, 300
    flow = np.zeros((H, W, 2))
    flow[..., 0] = 7.0                                     # constant dx
    flow[..., 1] = -3.0                                    # constant dy
    boxes = np.array([[50.0, 50, 120, 140], [10, 10, 40, 40]])
    warped = warp_boxes_by_flow(boxes, flow)
    assert np.allclose(warped, boxes + np.array([7, -3, 7, -3]), atol=1e-6)
    # identity (zero) flow is a no-op
    assert np.allclose(warp_boxes_by_flow(boxes, np.zeros((H, W, 2))), boxes, atol=1e-6)


def test_warp_masks_constant_flow_shifts():
    pytest.importorskip("torch")
    H, W = 64, 64
    mask = np.zeros((H, W), dtype=np.float32)
    mask[20:40, 20:40] = 1.0
    flow = np.zeros((H, W, 2), dtype=np.float32)
    flow[..., 0] = 10.0                                    # shift content +10 in x
    warped = warp_masks_by_flow(mask[None], flow, device="cpu")[0] > 0.5
    # the block's center of mass moves right by ~10 px
    ys, xs = np.where(warped)
    assert xs.mean() > 27 and abs(ys.mean() - 29.5) < 2    # y roughly unchanged, x shifted right
