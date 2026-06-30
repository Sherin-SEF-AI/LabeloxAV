"""M3: cross-sensor 2D-3D consistency. A 3D cuboid must reproject onto the 2D box of the same object in the
cameras that see it; a gross mismatch (best IoU below the floor in every camera) is flagged for review. The
check is conservative by design because the fleet runs on nominal calibration. These drive the pure scorer
against a real cam_f projection so the matching box is exactly the projection and the logic is deterministic."""

from __future__ import annotations

from services.lidar.quality3d.checker import _iou_2d, _projected_aabb, check_2d3d_consistency

W, H = 1280, 960
CUB = {"center": [8.0, 0.0, 0.0], "dims": [4.0, 2.0, 1.5], "yaw": 0.0, "pitch": 0.0, "roll": 0.0}


def test_iou_2d_basic():
    assert _iou_2d([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert _iou_2d([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_a_front_cuboid_projects_into_cam_f():
    box = _projected_aabb(CUB, "cam_f", W, H)
    assert box is not None and box[2] > box[0] and box[3] > box[1]


def test_consistent_when_the_2d_box_matches_the_projection():
    box = _projected_aabb(CUB, "cam_f", W, H)
    views = [{"cam_id": "cam_f", "w": W, "h": H, "bbox_2d": box}]
    assert check_2d3d_consistency(CUB, views, 0.3) is None        # IoU 1.0 -> consistent


def test_flagged_when_the_2d_box_is_far_from_the_projection():
    views = [{"cam_id": "cam_f", "w": W, "h": H, "bbox_2d": [4, 4, 44, 44]}]  # a corner box, nowhere near it
    flag = check_2d3d_consistency(CUB, views, 0.3)
    assert flag is not None and flag["kind"] == "box_2d3d_inconsistent"
    assert flag["detail"]["best_iou"] < 0.3
    assert flag["score"] > 0.0


def test_consistent_if_any_camera_agrees():
    box = _projected_aabb(CUB, "cam_f", W, H)
    # one matching view and one disagreeing view: the best IoU clears the floor, so it stays consistent
    views = [{"cam_id": "cam_f", "w": W, "h": H, "bbox_2d": box},
             {"cam_id": "cam_f", "w": W, "h": H, "bbox_2d": [0, 0, 8, 8]}]
    assert check_2d3d_consistency(CUB, views, 0.3) is None


def test_nothing_to_check_returns_none():
    assert check_2d3d_consistency(CUB, [], 0.3) is None
    assert check_2d3d_consistency(CUB, [{"cam_id": "cam_f", "w": W, "h": H, "bbox_2d": None}], 0.3) is None
