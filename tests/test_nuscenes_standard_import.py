"""A standard nuScenes table directory imports with its 3D boxes placed on the camera images.

Pure: a small scene is written to tmp_path as real nuScenes tables and read back through the import adapter.
No devkit, no infra.

Every expected number here is derived in this file, not by the code under test. Boxes are defined in a
camera's own optical frame (x right, y down, z forward), where what the image should show is plain
arithmetic: u = fx * x / z + cx. They are then carried out to the global frame with rotation MATRICES
composed from elementary rotations, and only the resulting poses are written into the tables as nuScenes
quaternions. The adapter has to come back the other way (quaternion -> matrix, global -> ego -> sensor), so
a wrong quaternion order, a transposed rotation, a transform applied in the wrong order or a swapped
width/length each land the box somewhere else than where this file put it.
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

import numpy as np
import pytest

from services.imports import adapter_nuscenes
from services.imports.adapter_nuscenes import project_corners_to_bbox
from services.imports.nuscenes_tables import quat_multiply, quat_to_matrix
from services.lidar.ingest.nuscenes_sweeps import (
    iter_nuscenes_lidar_clouds,
    list_nuscenes_lidar_sweeps,
)
from services.lidar.ingest.readers import read_point_cloud

W, H = 1600, 900
FX, FY, CX, CY = 1000.0, 1100.0, 800.0, 450.0
K = [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]]


# --- this file's own geometry (independent of the code under test) ----------------------------------------

def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _quat(r) -> list[float]:
    """[w, x, y, z] of a rotation matrix (Shepperd's method). The code under test never converts a matrix
    to a quaternion, so this shares nothing with it."""
    t = r[0, 0] + r[1, 1] + r[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    return [float(v) for v in q]


def _hom(r, t) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = t
    return m


# An optical camera frame looking along the ego's +x: camera x (right) is ego -y, camera y (down) is ego
# -z, camera z (forward) is ego +x. The columns are the camera axes written in the ego frame.
OPTICAL_TO_EGO = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

# Sensor mounts (sensor -> ego). Deliberately not axis-aligned: small roll/pitch on top of the yaw.
R_CAM_FRONT = _rz(0.3) @ _ry(-0.05) @ _rx(0.02) @ OPTICAL_TO_EGO
T_CAM_FRONT = np.array([1.7, 0.1, 1.5])
R_CAM_BACK = _rz(math.pi - 0.2) @ _rx(0.03) @ OPTICAL_TO_EGO
T_CAM_BACK = np.array([0.0, -0.05, 1.55])
R_LIDAR = _rz(-math.pi / 2 + 0.01) @ _rx(0.005)
T_LIDAR = np.array([0.94, 0.0, 1.84])

# Ego poses (ego -> global), one per sample_data row as in the real dataset: the cameras of one sample
# fire at different instants, so each has its own pose.
R_EGO_1F = _rz(1.1) @ _ry(0.04) @ _rx(-0.03)
T_EGO_1F = np.array([410.5, 1180.25, 0.3])
R_EGO_1B = _rz(1.102) @ _ry(0.04) @ _rx(-0.03)
T_EGO_1B = np.array([410.62, 1180.45, 0.3])
R_EGO_2F = _rz(1.15) @ _ry(0.03) @ _rx(-0.02)
T_EGO_2F = np.array([413.1, 1184.9, 0.32])
R_EGO_1L = _rz(1.101) @ _ry(0.04) @ _rx(-0.03)
T_EGO_1L = np.array([410.58, 1180.38, 0.3])

TS_SAMPLE_1 = 1532402927647951          # microseconds, as nuScenes stores them
TS_SAMPLE_2 = TS_SAMPLE_1 + 500_000
TS_CAM_FRONT_1 = 1532402927612460
TS_CAM_BACK_1 = 1532402927637525
TS_CAM_FRONT_2 = TS_CAM_FRONT_1 + 500_000


def _box_axes_in_camera(yaw: float) -> np.ndarray:
    """Rotation (columns = box x, y, z axes) of a box standing upright in front of a camera, written in the
    camera's optical frame. At yaw 0 its length points away from the camera (+z), its width to the
    camera's left (-x) and its height up (-y); `yaw` turns it about its own up axis."""
    ax = math.cos(yaw) * np.array([0, 0, 1.0]) + math.sin(yaw) * np.array([-1.0, 0, 0])
    ay = -math.sin(yaw) * np.array([0, 0, 1.0]) + math.cos(yaw) * np.array([-1.0, 0, 0])
    az = np.array([0, -1.0, 0])
    return np.stack([ax, ay, az], axis=1)


def _corners(center, axes, w, l, h) -> np.ndarray:  # noqa: E741
    """All 8 corners: centre +- half the length along the box x axis, width along y, height along z."""
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                out.append(np.asarray(center) + sx * l / 2 * axes[:, 0] + sy * w / 2 * axes[:, 1]
                           + sz * h / 2 * axes[:, 2])
    return np.array(out)


def _pixels(points_cam) -> np.ndarray:
    return np.array([[FX * x / z + CX, FY * y / z + CY] for x, y, z in points_cam])


# --- the scene, written as real nuScenes tables -----------------------------------------------------------

class _Scene:
    """Builds the table rows. Boxes are added in a camera's frame and stored in the global frame."""

    def __init__(self) -> None:
        self.t: dict[str, list] = {n: [] for n in (
            "category", "attribute", "visibility", "sensor", "calibrated_sensor", "ego_pose", "log", "scene",
            "sample", "sample_data", "instance", "sample_annotation")}
        self.t["visibility"] = [{"token": str(i), "level": lvl, "description": ""} for i, lvl in
                                enumerate(("v0-40", "v40-60", "v60-80", "v80-100"), start=1)]
        self.t["log"].append({"token": "log1", "logfile": "n015", "vehicle": "n015",
                              "date_captured": "2018-07-24", "location": "singapore-onenorth"})
        self.t["scene"].append({"token": "scene1", "name": "scene-0001", "description": "", "log_token": "log1",
                                "nbr_samples": 2, "first_sample_token": "s1", "last_sample_token": "s2"})
        self.t["sample"] += [
            {"token": "s1", "timestamp": TS_SAMPLE_1, "scene_token": "scene1", "prev": "", "next": "s2"},
            {"token": "s2", "timestamp": TS_SAMPLE_2, "scene_token": "scene1", "prev": "s1", "next": ""},
        ]
        self._cs: dict[str, str] = {}

    def sensor(self, channel, modality, r, t, intrinsic):
        self.t["sensor"].append({"token": f"sensor_{channel}", "channel": channel, "modality": modality})
        self.t["calibrated_sensor"].append({
            "token": f"cs_{channel}", "sensor_token": f"sensor_{channel}", "translation": list(map(float, t)),
            "rotation": _quat(r), "camera_intrinsic": intrinsic})
        self._cs[channel] = f"cs_{channel}"

    def data(self, token, sample, channel, ts, r_ego, t_ego, filename, key=True, dims=True):
        self.t["ego_pose"].append({"token": f"ego_{token}", "timestamp": ts, "rotation": _quat(r_ego),
                                   "translation": list(map(float, t_ego))})
        row = {"token": token, "sample_token": sample, "ego_pose_token": f"ego_{token}",
               "calibrated_sensor_token": self._cs[channel], "timestamp": ts, "fileformat": "jpg",
               "is_key_frame": key, "height": H if dims else 0, "width": W if dims else 0,
               "filename": filename, "prev": "", "next": ""}
        self.t["sample_data"].append(row)

    def category(self, name):
        if not any(c["name"] == name for c in self.t["category"]):
            self.t["category"].append({"token": f"cat_{name}", "name": name, "description": ""})
        return f"cat_{name}"

    def attribute(self, name):
        if not any(a["name"] == name for a in self.t["attribute"]):
            self.t["attribute"].append({"token": f"attr_{name}", "name": name, "description": ""})
        return f"attr_{name}"

    def box(self, token, sample, instance, category, center_cam, axes_cam, wlh, cam_r, cam_t, ego_r, ego_t,
            visibility="4", attributes=()):
        """Place a box given in a camera's optical frame into the global frame, by matrices."""
        if not any(i["token"] == instance for i in self.t["instance"]):
            self.t["instance"].append({"token": instance, "category_token": self.category(category),
                                       "nbr_annotations": 1, "first_annotation_token": token,
                                       "last_annotation_token": token})
        center_global = ego_r @ (cam_r @ np.asarray(center_cam, dtype=np.float64) + cam_t) + ego_t
        rot_global = ego_r @ cam_r @ axes_cam
        self.t["sample_annotation"].append({
            "token": token, "sample_token": sample, "instance_token": instance,
            "visibility_token": visibility, "attribute_tokens": [self.attribute(a) for a in attributes],
            "translation": [float(v) for v in center_global], "size": [float(v) for v in wlh],
            "rotation": _quat(rot_global), "prev": "", "next": "", "num_lidar_pts": 37, "num_radar_pts": 2})

    def write(self, dataroot: Path, version: str = "v1.0-mini") -> Path:
        d = dataroot / version
        d.mkdir(parents=True, exist_ok=True)
        for name, rows in self.t.items():
            (d / f"{name}.json").write_text(json.dumps(rows))
        for sd in self.t["sample_data"]:
            p = dataroot / sd["filename"]
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.write_bytes(b"\xff\xd8\xff\xd9")       # a file to point at; never decoded, dims are stated
        return dataroot


# The boxes, each given in CAM_FRONT's frame at the instant of the sample's CAM_FRONT image.
A_CENTER, A_YAW, A_WLH = (1.0, 0.5, 10.0), 0.4, (1.9, 4.5, 1.6)          # a car, fully in view, turned
B_CENTER, B_WLH = (-3.5, 0.3, -12.0), (2.0, 4.2, 1.5)                    # behind the front camera
C_CENTER, C_WLH = (7.5, 0.0, 10.0), (2.0, 4.0, 1.5)                      # runs off the right edge
D_CENTER, D_WLH = (0.0, 0.0, 1.0), (2.0, 4.0, 1.0)                       # straddles the camera plane
E_CENTER, E_WLH = (30.0, 0.0, 10.0), (2.0, 4.0, 1.5)                     # in front, far outside the view
A2_CENTER, A2_YAW = (-2.0, 0.4, 15.0), -0.3                              # the car again, one sample later


def _build_scene() -> _Scene:
    s = _Scene()
    s.sensor("CAM_FRONT", "camera", R_CAM_FRONT, T_CAM_FRONT, K)
    s.sensor("CAM_BACK", "camera", R_CAM_BACK, T_CAM_BACK, K)
    s.sensor("LIDAR_TOP", "lidar", R_LIDAR, T_LIDAR, [])

    s.data("sd_f1", "s1", "CAM_FRONT", TS_CAM_FRONT_1, R_EGO_1F, T_EGO_1F,
           f"samples/CAM_FRONT/n015__CAM_FRONT__{TS_CAM_FRONT_1}.jpg")
    s.data("sd_b1", "s1", "CAM_BACK", TS_CAM_BACK_1, R_EGO_1B, T_EGO_1B,
           f"samples/CAM_BACK/n015__CAM_BACK__{TS_CAM_BACK_1}.jpg")
    s.data("sd_f2", "s2", "CAM_FRONT", TS_CAM_FRONT_2, R_EGO_2F, T_EGO_2F,
           f"samples/CAM_FRONT/n015__CAM_FRONT__{TS_CAM_FRONT_2}.jpg")
    # A camera sweep between the two samples: unannotated, must not become a frame.
    s.data("sd_f_sweep", "s2", "CAM_FRONT", TS_CAM_FRONT_1 + 80_000, R_EGO_1F, T_EGO_1F,
           f"sweeps/CAM_FRONT/n015__CAM_FRONT__{TS_CAM_FRONT_1 + 80_000}.jpg", key=False)
    # LiDAR: the keyframe of sample 1 and one sweep after it.
    s.data("sd_l1", "s1", "LIDAR_TOP", TS_SAMPLE_1, R_EGO_1L, T_EGO_1L,
           f"samples/LIDAR_TOP/n015__LIDAR_TOP__{TS_SAMPLE_1}.pcd.bin")
    s.data("sd_l_sweep", "s2", "LIDAR_TOP", TS_SAMPLE_1 + 50_000, R_EGO_1B, T_EGO_1B,
           f"sweeps/LIDAR_TOP/n015__LIDAR_TOP__{TS_SAMPLE_1 + 50_000}.pcd.bin", key=False)

    front1 = (R_CAM_FRONT, T_CAM_FRONT, R_EGO_1F, T_EGO_1F)
    s.box("ann_a", "s1", "inst_car", "vehicle.car", A_CENTER, _box_axes_in_camera(A_YAW), A_WLH, *front1,
          visibility="4", attributes=("vehicle.moving",))
    s.box("ann_b", "s1", "inst_truck", "vehicle.truck", B_CENTER, _box_axes_in_camera(0.0), B_WLH, *front1,
          visibility="2", attributes=("vehicle.parked",))
    s.box("ann_c", "s1", "inst_bus", "vehicle.bus.rigid", C_CENTER, _box_axes_in_camera(0.0), C_WLH, *front1)
    s.box("ann_d", "s1", "inst_barrier", "movable_object.barrier", D_CENTER, _box_axes_in_camera(0.0), D_WLH,
          *front1)
    s.box("ann_e", "s1", "inst_ped", "human.pedestrian.adult", E_CENTER, _box_axes_in_camera(0.0), E_WLH,
          *front1)
    s.box("ann_a2", "s2", "inst_car", "vehicle.car", A2_CENTER, _box_axes_in_camera(A2_YAW), A_WLH,
          R_CAM_FRONT, T_CAM_FRONT, R_EGO_2F, T_EGO_2F, visibility="3")
    return s


@pytest.fixture
def dataroot(tmp_path) -> Path:
    return _build_scene().write(tmp_path / "nuscenes")


def _frames(root: Path, **kw) -> dict[tuple[str, int], object]:
    return {(f.cam_id, f.ts_ns): f for f in adapter_nuscenes.parse(root, **kw)}


def _by_token(frame) -> dict[str, object]:
    return {o.provenance["nuscenes"]["annotation_token"]: o for o in frame.objects}


def _same_rotation(q, r_expected, tol=1e-6) -> bool:
    """A quaternion the adapter reported names the expected rotation: compared through this file's own
    matrix -> quaternion conversion, up to the sign ambiguity of quaternions."""
    q, e = np.asarray(q, dtype=np.float64), np.asarray(_quat(r_expected))
    return bool(min(np.abs(q - e).max(), np.abs(q + e).max()) < tol)


# --- the math, against rotations written out by hand ------------------------------------------------------

def test_quaternion_convention_is_w_first_and_active():
    s = math.sqrt(0.5)
    # A quarter turn about z takes x to y; about x takes y to z; about y takes z to x.
    np.testing.assert_allclose(quat_to_matrix([s, 0, 0, s]), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose(quat_to_matrix([s, s, 0, 0]), [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-12)
    np.testing.assert_allclose(quat_to_matrix([s, 0, s, 0]), [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-12)
    # Not unit length in the file: normalized, not rejected and not scaled into a shear.
    np.testing.assert_allclose(quat_to_matrix([2 * s, 0, 0, 2 * s]), [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                               atol=1e-12)
    # The product applies the right-hand rotation first: Rz(90) @ Rx(90) as matrices.
    q = quat_multiply([s, 0, 0, s], [s, s, 0, 0])
    np.testing.assert_allclose(quat_to_matrix(q), _rz(math.pi / 2) @ _rx(math.pi / 2), atol=1e-12)
    for r in (R_CAM_FRONT, R_CAM_BACK, R_EGO_1F, R_LIDAR, R_EGO_1F @ R_CAM_BACK):
        np.testing.assert_allclose(quat_to_matrix(_quat(r)), r, atol=1e-12)
    with pytest.raises(ValueError):
        quat_to_matrix([0, 0, 0, 0])


# --- frames -----------------------------------------------------------------------------------------------

def test_one_frame_per_camera_keyframe_with_channel_time_and_image(dataroot):
    frames = adapter_nuscenes.parse(dataroot)
    assert [(f.cam_id, f.ts_ns) for f in frames] == [
        ("CAM_BACK", TS_SAMPLE_1 * 1000), ("CAM_FRONT", TS_SAMPLE_1 * 1000), ("CAM_FRONT", TS_SAMPLE_2 * 1000)]
    front1 = next(f for f in frames if f.cam_id == "CAM_FRONT" and f.ts_ns == TS_SAMPLE_1 * 1000)
    assert front1.image_ref == f"samples/CAM_FRONT/n015__CAM_FRONT__{TS_CAM_FRONT_1}.jpg"
    assert (dataroot / front1.image_ref).is_file()
    assert (front1.width, front1.height) == (W, H)
    # Neither the unannotated camera sweep nor the LiDAR rows became frames.
    assert not any("sweeps/" in f.image_ref or "LIDAR" in f.image_ref for f in frames)


def test_capture_time_is_available_instead_of_the_sample_time(dataroot):
    frames = adapter_nuscenes.parse(dataroot, frame_time="capture")
    assert sorted((f.cam_id, f.ts_ns) for f in frames) == [
        ("CAM_BACK", TS_CAM_BACK_1 * 1000), ("CAM_FRONT", TS_CAM_FRONT_1 * 1000),
        ("CAM_FRONT", TS_CAM_FRONT_2 * 1000)]
    with pytest.raises(ValueError):
        adapter_nuscenes.parse(dataroot, frame_time="lidar")


def test_image_ref_stays_inside_a_root_that_wraps_the_dataset(dataroot):
    # An extracted archive often has the dataset one folder down. The reference must still be a path
    # under the import root, which is the only place the import driver will look.
    wrapper = dataroot.parent
    frames = adapter_nuscenes.parse(wrapper)
    assert all((wrapper / f.image_ref).is_file() for f in frames)
    assert all(f.image_ref.startswith("nuscenes/samples/") for f in frames)


# --- the 3D box on the image ------------------------------------------------------------------------------

def test_a_box_at_a_known_pose_lands_where_arithmetic_says(dataroot):
    frame = _frames(dataroot)[("CAM_FRONT", TS_SAMPLE_1 * 1000)]
    obj = _by_token(frame)["ann_a"]

    axes = _box_axes_in_camera(A_YAW)
    w, l, h = A_WLH  # noqa: E741
    corners = _corners(A_CENTER, axes, w, l, h)
    assert (corners[:, 2] > 5).all()                              # all in front: plain projection is valid
    px = _pixels(corners)
    assert px[:, 0].min() > 0 and px[:, 0].max() < W and px[:, 1].min() > 0 and px[:, 1].max() < H
    expected = [px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()]
    assert obj.bbox == pytest.approx(expected, abs=0.01)
    # A swapped width/length would give a visibly different box, so this is not satisfied by accident.
    swapped = _pixels(_corners(A_CENTER, axes, l, w, h))
    assert abs((swapped[:, 0].max() - swapped[:, 0].min()) - (expected[2] - expected[0])) > 20

    assert obj.name == "vehicle.car"
    assert obj.track_ref == "inst_car"
    assert obj.conf == 1.0
    assert obj.attrs == {"nuscenes_visibility": "v80-100", "nuscenes_attributes": ["vehicle.moving"]}
    meta = obj.provenance["nuscenes"]
    assert meta["instance_token"] == "inst_car" and meta["sample_token"] == "s1"
    assert meta["sample_data_token"] == "sd_f1" and meta["channel"] == "CAM_FRONT"
    assert meta["capture_timestamp_us"] == TS_CAM_FRONT_1 and meta["scene_name"] == "scene-0001"
    assert (meta["num_lidar_pts"], meta["num_radar_pts"]) == (37, 2)


def test_the_cuboid_is_carried_in_the_sensor_ego_and_global_frames(dataroot):
    obj = _by_token(_frames(dataroot)[("CAM_FRONT", TS_SAMPLE_1 * 1000)])["ann_a"]
    axes = _box_axes_in_camera(A_YAW)
    box = obj.provenance["nuscenes_box"]

    # Sensor frame: exactly the pose the box was defined at.
    assert box["sensor"]["channel"] == "CAM_FRONT"
    assert box["sensor"]["center"] == pytest.approx(list(A_CENTER), abs=1e-4)
    assert _same_rotation(box["sensor"]["rotation"], axes)
    assert box["size_wlh"] == pytest.approx(list(A_WLH))

    # Ego frame: one step out, by this file's matrices.
    center_ego = R_CAM_FRONT @ np.array(A_CENTER) + T_CAM_FRONT
    axes_ego = R_CAM_FRONT @ axes
    assert box["ego"]["center"] == pytest.approx(center_ego.tolist(), abs=1e-4)
    assert _same_rotation(box["ego"]["rotation"], axes_ego)

    # Global frame: what the table said.
    center_global = R_EGO_1F @ center_ego + T_EGO_1F
    assert box["global"]["center"] == pytest.approx(center_global.tolist(), abs=1e-4)
    assert _same_rotation(box["global"]["rotation"], R_EGO_1F @ axes_ego)

    # And in the shape Object.cuboid_3d has: ego frame, size [width, length, height], yaw of the length
    # axis about ego up.
    cuboid = obj.provenance["cuboid_3d"]
    assert set(cuboid) == {"center", "size", "yaw"}
    assert cuboid["center"] == pytest.approx(center_ego.tolist(), abs=1e-4)
    assert cuboid["size"] == pytest.approx(list(A_WLH))
    length_axis = axes_ego[:, 0]
    assert cuboid["yaw"] == pytest.approx(math.atan2(length_axis[1], length_axis[0]), abs=1e-6)
    # The mount is yawed 0.3 rad and the box 0.4 rad to the camera's left, so its heading in the ego frame
    # is close to their sum; the small roll/pitch of the mount accounts for the rest.
    assert cuboid["yaw"] == pytest.approx(0.7, abs=0.03)


def test_a_box_behind_one_camera_is_an_object_only_of_the_camera_that_sees_it(dataroot):
    frames = _frames(dataroot)
    assert "ann_b" not in _by_token(frames[("CAM_FRONT", TS_SAMPLE_1 * 1000)])

    back = _by_token(frames[("CAM_BACK", TS_SAMPLE_1 * 1000)])
    assert set(back) == {"ann_b"}                                # nothing else is behind the vehicle
    obj = back["ann_b"]

    # Where the rear camera should see it, through homogeneous matrices and a general inverse: defined in
    # the front camera at its instant, out to global, back in through the rear camera's own ego pose.
    w, l, h = B_WLH  # noqa: E741
    corners_front = _corners(B_CENTER, _box_axes_in_camera(0.0), w, l, h)
    front_to_global = _hom(R_EGO_1F, T_EGO_1F) @ _hom(R_CAM_FRONT, T_CAM_FRONT)
    global_to_back = np.linalg.inv(_hom(R_EGO_1B, T_EGO_1B) @ _hom(R_CAM_BACK, T_CAM_BACK))
    hom = np.c_[corners_front, np.ones(8)] @ (global_to_back @ front_to_global).T
    corners_back = hom[:, :3]
    assert (corners_back[:, 2] > 5).all()
    px = _pixels(corners_back)
    assert px[:, 0].min() > 0 and px[:, 0].max() < W and px[:, 1].min() > 0 and px[:, 1].max() < H
    assert obj.bbox == pytest.approx([px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()], abs=0.01)

    assert obj.name == "vehicle.truck" and obj.track_ref == "inst_truck"
    assert obj.attrs == {"nuscenes_visibility": "v40-60", "nuscenes_attributes": ["vehicle.parked"]}
    assert obj.provenance["nuscenes_box"]["sensor"]["channel"] == "CAM_BACK"
    center_back = (np.r_[B_CENTER, 1.0] @ (global_to_back @ front_to_global).T)[:3]
    assert obj.provenance["nuscenes_box"]["sensor"]["center"] == pytest.approx(center_back.tolist(), abs=1e-4)


def test_a_box_running_off_the_image_is_clipped_along_its_silhouette(dataroot):
    obj = _by_token(_frames(dataroot)[("CAM_FRONT", TS_SAMPLE_1 * 1000)])["ann_c"]
    # Axis-aligned in the camera: x in [6.5, 8.5], y in [-0.75, 0.75], z in [8, 12].
    # Left edge: the far-left corners, u = 800 + 1000 * 6.5 / 12.
    x1 = CX + FX * 6.5 / 12
    # The near-left corners sit just past the right edge (u = 800 + 1000 * 6.5 / 8 = 1612.5), so the top of
    # what is visible is where the silhouette's top edge, from far-left-top to near-left-top, crosses
    # u = 1600, not the height of the near corners themselves.
    u_far, u_near = CX + FX * 6.5 / 12, CX + FX * 6.5 / 8
    assert u_far < W < u_near
    v_far, v_near = FY * 0.75 / 12, FY * 0.75 / 8                 # half-heights about the principal row
    half = v_far + (W - u_far) / (u_near - u_far) * (v_near - v_far)
    assert obj.bbox == pytest.approx([x1, CY - half, W, CY + half], abs=0.01)
    # Clamping the box of all eight projected corners would have reported the off-image near face's height.
    assert CY - half - (CY - v_near) > 1.0
    assert obj.bbox[2] == W                                       # exactly the image edge, not 1599.99


def test_a_box_straddling_the_camera_plane_is_cut_at_the_near_plane(dataroot):
    obj = _by_token(_frames(dataroot)[("CAM_FRONT", TS_SAMPLE_1 * 1000)])["ann_d"]
    # The box spans z in [-1, 3] around the camera, 2 m wide and 1 m high. Cut at the near plane its front
    # face is 0.1 m from the lens and far larger than the view (u = 800 +- 1000 * 1 / 0.1), so the visible
    # part fills the image. Projecting only the four corners in front of the camera (z = 3) would report
    # u in [467, 1133], and projecting the rear ones through the lens would mirror them.
    assert obj.bbox == pytest.approx([0, 0, W, H], abs=0.01)
    far_only = [CX - FX * 1 / 3, CY - FY * 0.5 / 3, CX + FX * 1 / 3, CY + FY * 0.5 / 3]
    assert far_only[0] > 400 and far_only[2] < 1200


def test_boxes_outside_every_view_are_not_objects(dataroot):
    seen = {tok for f in adapter_nuscenes.parse(dataroot) for tok in _by_token(f)}
    assert "ann_e" not in seen                                    # in front of CAM_FRONT but far to the side
    assert seen == {"ann_a", "ann_b", "ann_c", "ann_d", "ann_a2"}


def test_the_instance_token_links_one_object_across_samples(dataroot):
    frames = _frames(dataroot)
    first = _by_token(frames[("CAM_FRONT", TS_SAMPLE_1 * 1000)])["ann_a"]
    second = _by_token(frames[("CAM_FRONT", TS_SAMPLE_2 * 1000)])["ann_a2"]
    assert first.track_ref == second.track_ref == "inst_car"
    assert second.attrs == {"nuscenes_visibility": "v60-80"}     # no attribute tokens: no empty list

    # The second sample has its own ego pose; the box must be placed with that one, not the first.
    corners = _corners(A2_CENTER, _box_axes_in_camera(A2_YAW), *A_WLH)
    px = _pixels(corners)
    assert second.bbox == pytest.approx([px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()],
                                        abs=0.01)
    assert second.provenance["nuscenes_box"]["sensor"]["center"] == pytest.approx(list(A2_CENTER), abs=1e-4)


# --- the projection on its own ----------------------------------------------------------------------------

def _plate(center_px, half_diagonal_px, z=10.0, f=1000.0):
    """A square plate facing the camera, turned 45 degrees in the image plane, so its silhouette is a
    diamond |du| + |dv| <= half_diagonal_px around center_px. It is 1 mm thick, which moves a corner
    800 px off-axis by 0.04 px. Returns its corners and the intrinsic."""
    side = half_diagonal_px * math.sqrt(2) * z / f
    c = math.sqrt(0.5)
    axes = np.array([[c, -c, 0], [c, c, 0], [0, 0, 1.0]])
    center = [(center_px[0] - 800.0) * z / f, (center_px[1] - 450.0) * z / f, z]
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(np.array(center) + sx * side / 2 * axes[:, 0] + sy * side / 2 * axes[:, 1]
                               + sz * 0.0005 * axes[:, 2])
    return np.array(corners), [[f, 0, 800.0], [0, f, 450.0], [0, 0, 1]]


def test_a_silhouette_that_only_crosses_an_image_corner():
    # Diamond of half-diagonal 80 px centred 10 px inside the top-right corner region: within the image
    # it reaches left to u = 1510 (at v = 10) and down to v = 90 (at u = 1590).
    corners, k = _plate((1590.0, 10.0), 80.0)
    assert project_corners_to_bbox(corners, k, W, H) == pytest.approx([1510, 0, 1600, 90], abs=0.1)

    # The same diamond centred outside the corner at (1650, -50): its bounding box [1570, 1730] x
    # [-130, 30] overlaps the image, but the nearest image point (1600, 0) is |50| + |50| = 100 > 80 away
    # in the diamond's own metric, so none of it is visible.
    corners, k = _plate((1650.0, -50.0), 80.0)
    assert project_corners_to_bbox(corners, k, W, H) is None


def test_projection_rejects_what_cannot_be_seen():
    k = [[1000.0, 0, 800.0], [0, 1000.0, 450.0], [0, 0, 1]]
    behind, _ = _plate((800.0, 450.0), 80.0, z=-10.0)
    assert project_corners_to_bbox(behind, k, W, H) is None
    speck, _ = _plate((800.0, 450.0), 0.2)                        # 0.4 px across: no visible area
    assert project_corners_to_bbox(speck, k, W, H) is None
    seen, _ = _plate((800.0, 450.0), 80.0)
    assert project_corners_to_bbox(seen, k, W, H) == pytest.approx([720, 370, 880, 530], abs=0.1)


# --- degraded inputs --------------------------------------------------------------------------------------

def test_unusable_annotations_and_uncalibrated_cameras_do_not_sink_the_import(tmp_path):
    s = _build_scene()
    s.t["sample_annotation"][0]["rotation"] = [0, 0, 0, 0]        # ann_a: not a rotation
    s.t["sample_annotation"][2]["size"] = [0, 0, 0]               # ann_c: a placeholder box
    next(c for c in s.t["calibrated_sensor"] if c["token"] == "cs_CAM_BACK")["camera_intrinsic"] = []
    del s.t["attribute"][:]                                       # optional table, emptied
    root = s.write(tmp_path / "ds")
    (root / "v1.0-mini" / "visibility.json").unlink()             # optional table, absent

    frames = _frames(root)
    assert len(frames) == 3                                       # every keyframe is still a frame
    front = _by_token(frames[("CAM_FRONT", TS_SAMPLE_1 * 1000)])
    assert set(front) == {"ann_d"}
    assert front["ann_d"].attrs == {"nuscenes_visibility": "4"}   # the raw token when the table is gone
    assert frames[("CAM_BACK", TS_SAMPLE_1 * 1000)].objects == []


def test_a_missing_required_table_is_named(tmp_path):
    root = _build_scene().write(tmp_path / "ds")
    (root / "v1.0-mini" / "ego_pose.json").unlink()
    with pytest.raises(FileNotFoundError, match="ego_pose.json"):
        adapter_nuscenes.parse(root)
    with pytest.raises(FileNotFoundError):
        adapter_nuscenes.parse(tmp_path / "ds" / "samples")


# --- LabeloxAV's own export still imports as before -------------------------------------------------------

def _export_record(object_id, frame_id, ts, class_id, class_name, bbox, track_id=None, cuboid=None):
    from services.export.records import ExportRecord

    return ExportRecord(
        object_id=object_id, frame_id=frame_id, session_id=uuid.UUID(int=7), ts_ns=ts, cam_id="cam_f",
        img_uri=f"s3://frames/{frame_id}.jpg", width=640, height=480, vehicle_id="TIGOR-07", city="BLR",
        class_id=class_id, class_name=class_name, bbox=bbox, conf=0.83, state="accepted", source="human",
        mask_uri=None, track_id=track_id, attrs={"occlusion": 25}, cuboid_3d=cuboid)


def test_our_own_export_round_trips_as_it_did(tmp_path):
    from services.autolabel.ontology import get_ontology
    from services.export.adapter_nuscenes import write_nuscenes

    onto = get_ontology()
    sedan, ped = onto.by_name("sedan"), onto.by_name("pedestrian")
    f1, f2, track = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    cuboid = {"center": [12.0, -1.5, 0.8], "size": [1.8, 4.4, 1.5], "yaw": 0.35}
    recs = [
        _export_record(uuid.uuid4(), f1, 1_000_000_000, sedan.id, "sedan", [100.126, 50.0, 220.5, 140.0],
                       track_id=track, cuboid=cuboid),
        _export_record(uuid.uuid4(), f1, 1_000_000_000, ped.id, "pedestrian", [300.0, 60.0, 330.0, 150.0]),
        _export_record(uuid.uuid4(), f2, 2_000_000_000, sedan.id, "sedan", [110.0, 52.0, 230.0, 142.0],
                       track_id=track),
    ]
    write_nuscenes(recs, onto, tmp_path / "export" / "nuscenes")

    frames = adapter_nuscenes.parse(tmp_path / "export")
    assert len(frames) == 2
    by_ref = {f.image_ref: f for f in frames}
    a, b = by_ref[f"s3://frames/{f1}.jpg"], by_ref[f"s3://frames/{f2}.jpg"]
    # Exactly the frame fields this path has always produced.
    assert (a.width, a.height, a.ts_ns, a.cam_id) == (640, 480, 1_000_000_000, "cam_front")
    assert b.ts_ns == 2_000_000_000
    assert sorted((o.name, o.bbox) for o in a.objects) == [
        ("pedestrian", [300.0, 60.0, 330.0, 150.0]), ("sedan", [100.13, 50.0, 220.5, 140.0])]
    assert all(o.conf == 0.83 and o.attrs == {} for o in a.objects)
    sedan_a = next(o for o in a.objects if o.name == "sedan")
    assert sedan_a.track_ref is not None and sedan_a.track_ref == b.objects[0].track_ref
    # The real cuboid the export wrote comes back in Object.cuboid_3d shape; a placeholder does not.
    got = sedan_a.provenance["cuboid_3d"]
    assert got["center"] == cuboid["center"] and got["size"] == cuboid["size"]
    assert got["yaw"] == pytest.approx(0.35, abs=1e-5)
    assert next(o for o in a.objects if o.name == "pedestrian").provenance == {}
    assert b.objects[0].provenance == {}


def test_an_empty_export_of_ours_is_not_mistaken_for_a_standard_release(tmp_path):
    from services.autolabel.ontology import get_ontology
    from services.export.adapter_nuscenes import write_nuscenes

    write_nuscenes([], get_ontology(), tmp_path / "nuscenes")
    assert adapter_nuscenes.parse(tmp_path) == []


# --- LiDAR sweeps -----------------------------------------------------------------------------------------

def _write_sweep(path: Path, pts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pts.astype("<f4").tobytes())


def _lidar_points() -> np.ndarray:
    # 8 points (x, y, z, intensity 0..255, ring): 40 floats, divisible by both 4 and 5, as a real sweep's
    # float count is whenever its point count is a multiple of 4.
    rng = np.random.default_rng(11)
    xyz = rng.uniform(-30, 30, size=(8, 3))
    return np.c_[xyz, rng.uniform(0, 255, size=8), np.arange(8) % 32]


def test_lidar_sweeps_are_listed_with_their_transforms(dataroot):
    sweeps = list_nuscenes_lidar_sweeps(dataroot)
    assert [s.token for s in sweeps] == ["sd_l1", "sd_l_sweep"]
    key, sweep = sweeps
    assert key.path == dataroot / f"samples/LIDAR_TOP/n015__LIDAR_TOP__{TS_SAMPLE_1}.pcd.bin"
    assert key.ts_ns == TS_SAMPLE_1 * 1000 and key.is_key_frame and key.sample_token == "s1"
    assert key.sample_ts_ns == TS_SAMPLE_1 * 1000 and key.channel == "LIDAR_TOP"
    np.testing.assert_allclose(key.sensor_to_ego, _hom(R_LIDAR, T_LIDAR), atol=1e-12)
    np.testing.assert_allclose(key.ego_to_global, _hom(R_EGO_1L, T_EGO_1L), atol=1e-9)
    np.testing.assert_allclose(key.sensor_to_global, _hom(R_EGO_1L, T_EGO_1L) @ _hom(R_LIDAR, T_LIDAR),
                               atol=1e-9)
    # The sweep between keyframes has its own ego pose and its own time.
    assert not sweep.is_key_frame and sweep.ts_ns == (TS_SAMPLE_1 + 50_000) * 1000
    np.testing.assert_allclose(sweep.ego_to_global, _hom(R_EGO_1B, T_EGO_1B), atol=1e-9)

    assert [s.token for s in list_nuscenes_lidar_sweeps(dataroot, keyframes_only=True)] == ["sd_l1"]
    assert list_nuscenes_lidar_sweeps(dataroot, channel="LIDAR_FRONT") == []


def test_lidar_clouds_are_read_into_the_ego_frame(dataroot):
    pts = _lidar_points()
    key_path = dataroot / f"samples/LIDAR_TOP/n015__LIDAR_TOP__{TS_SAMPLE_1}.pcd.bin"
    _write_sweep(key_path, pts)
    (dataroot / f"sweeps/LIDAR_TOP/n015__LIDAR_TOP__{TS_SAMPLE_1 + 50_000}.pcd.bin").unlink()   # not downloaded

    got = list(iter_nuscenes_lidar_clouds(dataroot, keyframes_only=False))
    assert len(got) == 1                                          # the missing blob is skipped, not fatal
    sweep, cloud = got[0]
    assert cloud.frame == "ego" and cloud.ts_ns == TS_SAMPLE_1 * 1000 and cloud.n == 8
    expected = pts[:, :3] @ R_LIDAR.T + T_LIDAR
    np.testing.assert_allclose(cloud.xyz, expected, atol=1e-4)
    np.testing.assert_allclose(cloud.intensity, pts[:, 3] / 255.0, atol=1e-6)
    assert cloud.ring.tolist() == (np.arange(8) % 32).tolist()

    _, world = next(iter(iter_nuscenes_lidar_clouds(dataroot, frame="global")))
    np.testing.assert_allclose(world.xyz, expected @ R_EGO_1L.T + T_EGO_1L, atol=2e-3)
    _, raw = next(iter(iter_nuscenes_lidar_clouds(dataroot, frame="sensor")))
    np.testing.assert_allclose(raw.xyz, pts[:, :3], atol=1e-5)
    assert raw.frame == "nuscenes_lidar"


def test_a_nuscenes_sweep_file_is_read_as_five_floats_per_point(dataroot):
    pts = _lidar_points()
    path = dataroot / "samples/LIDAR_TOP/n015__LIDAR_TOP__1.pcd.bin"
    _write_sweep(path, pts)
    cloud = read_point_cloud(path)
    assert cloud.n == 8                                           # not 10 points of four floats
    np.testing.assert_allclose(cloud.xyz, pts[:, :3], atol=1e-5)
    # A KITTI file of the same size keeps its four-float reading.
    kitti = dataroot / "000001.bin"
    kitti.write_bytes(pts.astype("<f4").tobytes())
    assert read_point_cloud(kitti).n == 10
