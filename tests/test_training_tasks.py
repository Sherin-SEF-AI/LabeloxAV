"""Segmentation and pose are trainable, not just labelable.

The audit's largest capability asymmetry: the engine labels masks and keypoints, exports them, and (since the
metrics work) evaluates them, but `services/training/tasks/` held exactly one plugin, so neither could ever
improve a model. `tasks/base.py` even said so ("detection now; segmentation/classification later").

The label writers are the part most likely to be silently wrong, because a malformed line does not crash, it
trains a worse model. These tests pin the exact wire format each trainer expects."""
from __future__ import annotations

import pytest

from services.training.segmentation_dataset import SegBuildSpec, polygon_label_line
from services.training.tasks import get_task, list_tasks
from services.training.tasks.pose import SKELETONS, PoseBuildSpec, keypoint_label_line


# ---------------- registry ----------------

def test_all_three_task_types_are_registered():
    types = {t["task_type"] for t in list_tasks()}
    assert {"detection", "segmentation", "pose"} <= types


def test_each_task_starts_from_a_head_appropriate_checkpoint():
    # Fine-tuning a detection checkpoint for segmentation would train a mask head from scratch, which needs
    # far more data than a corpus of human-corrected masks holds.
    assert "seg" in get_task("segmentation").default_base_weights()
    assert "pose" in get_task("pose").default_base_weights()


def test_unknown_task_type_is_refused():
    with pytest.raises(ValueError, match="unknown task_type"):
        get_task("telepathy")


# ---------------- segmentation label format ----------------

def test_polygon_line_is_class_then_normalized_vertices():
    line = polygon_label_line(3, [0.0, 0.0, 100.0, 0.0, 100.0, 50.0], width=200, height=100)
    parts = line.split()
    assert parts[0] == "3"
    assert len(parts) == 1 + 6                       # three vertices, x and y each
    assert float(parts[1]) == 0.0 and float(parts[2]) == 0.0
    assert float(parts[3]) == pytest.approx(0.5)     # 100/200
    assert float(parts[6]) == pytest.approx(0.5)     # 50/100


def test_polygon_vertices_are_clamped_into_the_image():
    # An annotation dragged past the edge must not emit coordinates above 1.0: the loader would reject the
    # file, taking every other label in it down too.
    line = polygon_label_line(0, [-10.0, -10.0, 500.0, 500.0, 250.0, 10.0], width=200, height=100)
    coords = [float(v) for v in line.split()[1:]]
    assert all(0.0 <= c <= 1.0 for c in coords)


def test_a_degenerate_polygon_is_rejected():
    # Two points are a line, not a mask. Training on slivers teaches the model to emit slivers.
    assert polygon_label_line(0, [10.0, 10.0, 20.0, 20.0], width=100, height=100) is None


def test_segmentation_spec_inherits_the_safe_split_defaults():
    spec = SegBuildSpec()
    assert spec.group_split_by_session is True
    assert spec.exclude_gold_id is None
    assert spec.min_polygon_points == 3


# ---------------- pose label format ----------------

def _points(n: int, visibility: int = 2) -> list[list[float]]:
    return [[10.0 * i, 5.0 * i, visibility] for i in range(n)]


def test_keypoint_line_is_class_box_then_triplets():
    n = SKELETONS["person_17"]
    line = keypoint_label_line(1, [0.0, 0.0, 100.0, 100.0], _points(n), width=200, height=200)
    parts = line.split()
    assert parts[0] == "1"
    assert len(parts) == 1 + 4 + 3 * n               # class, box, then x y v per keypoint
    assert float(parts[1]) == pytest.approx(0.25)    # cx = 50/200
    assert float(parts[3]) == pytest.approx(0.5)     # bw = 100/200


def test_invisible_keypoints_are_written_at_the_origin_with_v_zero():
    # The loss skips v=0. Writing the stale coordinate instead would train the model toward a joint nobody
    # actually marked.
    pts = _points(17)
    pts[5] = [123.0, 456.0, 0]
    parts = keypoint_label_line(0, [0.0, 0.0, 10.0, 10.0], pts, width=1000, height=1000).split()
    kp5 = parts[5 + 3 * 5: 5 + 3 * 5 + 3]
    assert kp5 == ["0.000000", "0.000000", "0"]


def test_keypoint_coordinates_are_clamped():
    pts = [[-50.0, -50.0, 2]] + _points(16)
    parts = keypoint_label_line(0, [0.0, 0.0, 10.0, 10.0], pts, width=100, height=100).split()
    assert all(0.0 <= float(v) <= 1.0 for v in parts[5::3][:1])


def test_a_zero_area_box_is_rejected():
    assert keypoint_label_line(0, [10.0, 10.0, 10.0, 10.0], _points(17), 100, 100) is None


def test_pose_spec_defaults_to_the_editor_skeleton():
    spec = PoseBuildSpec()
    assert spec.skeleton in SKELETONS
    assert spec.group_split_by_session is True
    assert spec.min_visible_keypoints >= 1


def test_skeleton_point_count_is_pinned_not_inferred():
    # The count is part of the label layout (kpt_shape). Inferring it from whatever the first annotation
    # happened to contain would misalign every index in the dataset.
    assert SKELETONS["person_17"] == 17
