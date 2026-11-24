"""The 3D export's "waymo" output is a JSON file with Waymo's box field names, and says so.

Pure: the writer takes plain records and writes to tmp_path. It was named and documented as a Waymo export;
the Waymo Open Dataset format is TFRecord files of Frame protobufs, which nothing here produces. These
tests pin the honest name, the alias kept for existing callers, and the file describing itself.
"""

from __future__ import annotations

import json
import uuid

from services.lidar.export import adapters3d, write_waymo_3d, write_waymo_style_json_3d


def _rec(name: str, cloud_id: str, yaw: float) -> dict:
    return {"object_3d_id": str(uuid.uuid4()), "cloud_id": cloud_id, "class_name": name,
            "center": [10.0, 1.0, 0.75], "dims": [4.0, 1.8, 1.5], "yaw": yaw, "conf": 0.9}


def test_the_old_name_and_the_old_format_key_still_work():
    assert write_waymo_3d is write_waymo_style_json_3d
    assert adapters3d._FORMAT_WRITERS["waymo"] is adapters3d._FORMAT_WRITERS["waymo_json"]
    assert adapters3d._FORMAT_WRITERS["waymo_json"] is write_waymo_style_json_3d


def test_the_file_is_json_that_says_it_is_not_a_waymo_tfrecord(tmp_path):
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    path = write_waymo_style_json_3d([_rec("sedan", a, 0.2), _rec("pedestrian", a, -1.0), _rec("bus", b, 3.0)],
                                     tmp_path)
    assert path.name == "waymo_3d.json" and not list(tmp_path.glob("*.tfrecord*"))
    doc = json.loads(path.read_text())
    assert doc["format"] == "labelox.waymo_style_laser_labels_json/1"
    assert "Not a Waymo Open Dataset TFRecord" in doc["note"]

    # The layout older readers of this file relied on is unchanged: frames -> laser_labels -> box.
    frames = {f["context_name"]: f["laser_labels"] for f in doc["frames"]}
    assert set(frames) == {a, b} and len(frames[a]) == 2 and len(frames[b]) == 1
    label = frames[a][0]
    assert label["type"] == "sedan" and label["score"] == 0.9
    assert label["box"] == {"center_x": 10.0, "center_y": 1.0, "center_z": 0.75, "length": 4.0, "width": 1.8,
                            "height": 1.5, "heading": 0.2}


def test_the_documentation_does_not_claim_the_waymo_format():
    doc = write_waymo_style_json_3d.__doc__
    assert "NOT the Waymo Open Dataset format" in doc and "TFRecord" in doc
    assert "NOT the Waymo Open Dataset format" in adapters3d.__doc__
