"""SEC-M2: the moving-vs-static SceneModel fork.

Proves the AV pack exposes a moving-camera scene model and the Sec static-camera model is its opposite pole:
constant pose (ignores ego motion), no ground plane, is_static True. Uses a nominal calibration (config
defaults), no infra.
"""

from __future__ import annotations

import numpy as np

from core.geometry import Plane, se3, transform_points
from packs.av.scene_model import MovingCameraSceneModel, MovingCameraSceneModelFactory
from packs.base import SceneModel, SceneModelFactory
from packs.registry import get_pack
from packs.sec.scene_model import StaticCameraSceneModel, StaticCameraSceneModelFactory
from services.calibration.resolve import nominal_calibration


def _cal():
    return nominal_calibration("cam_f", 1920, 1080)


def test_av_pack_exposes_a_moving_scene_model_factory():
    factory = get_pack("av").scene_model
    assert isinstance(factory, SceneModelFactory)
    model = factory.build(_cal())
    assert isinstance(model, SceneModel)
    assert isinstance(model, MovingCameraSceneModel)


def test_moving_is_not_static_and_has_a_ground_plane():
    m = MovingCameraSceneModelFactory().build(_cal())
    assert m.is_static() is False
    gp = m.ground_plane()
    assert isinstance(gp, Plane)
    assert gp.normal == (0.0, 0.0, 1.0) and gp.frame == "ego"


def test_static_is_static_and_has_no_ground_plane():
    s = StaticCameraSceneModelFactory().build(_cal())
    assert isinstance(s, SceneModel)
    assert s.is_static() is True
    assert s.ground_plane() is None


def test_moving_camera_pose_composes_ego_motion_but_static_ignores_it():
    cal = _cal()
    moving = MovingCameraSceneModel(cal)
    static = StaticCameraSceneModel(cal)

    world_from_ego = np.eye(4)
    world_from_ego[:3, 3] = [10.0, -2.0, 0.5]  # the vehicle has driven 10 m forward, 2 m right, 0.5 m up

    # Moving: world<-camera shifts with the ego pose. Static: the same ego pose is ignored (fixed camera).
    assert np.allclose(moving.camera_pose(world_from_ego)[:3, 3] - moving.extrinsic()[:3, 3],
                       [10.0, -2.0, 0.5])
    assert np.allclose(static.camera_pose(world_from_ego), static.extrinsic())
    assert np.allclose(static.camera_pose(None), static.camera_pose(world_from_ego))


def test_extrinsic_matches_the_calibration_convention():
    # From cam = (ego - t) @ R the mount pose is p_ego = R @ p_cam + t; the extrinsic must reproduce that.
    cal = _cal()
    m = MovingCameraSceneModel(cal)
    expected = se3(cal.R().astype(np.float64), cal.t().astype(np.float64))
    assert np.allclose(m.extrinsic(), expected)
    # A camera-frame point maps into the ego frame by the extrinsic.
    p_cam = np.array([[0.0, 0.0, 5.0]])  # 5 m straight ahead along the optical axis
    p_ego = transform_points(m.extrinsic(), p_cam)[0]
    # Optical +z (forward) is ego +x (forward): the point should be ~5 m ahead of the camera mount.
    assert p_ego[0] > cal.t()[0] + 4.0
