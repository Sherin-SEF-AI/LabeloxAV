"""CVAT XML and Label Studio JSON interop.

These two formats are the migration path in and out of the tools most teams already run, so the thing worth
testing is that the import and export halves agree with each other. Both are pure file-level round trips: no
DB, no infra.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from services.imports.adapter_cvat import parse as cvat_parse
from services.imports.adapter_labelstudio import parse as ls_parse

# ---- CVAT ------------------------------------------------------------------------------------------------

CVAT_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <image id="0" name="frame_000.jpg" width="640" height="480">
    <box label="autorickshaw" xtl="10.0" ytl="20.0" xbr="110.0" ybr="140.0" rotation="15.5">
      <attribute name="overload">true</attribute>
    </box>
    <polygon label="pedestrian" points="10,10;60,10;60,80;10,80"/>
    <polyline label="curb" points="0,470;640,460"/>
    <points label="pose" points="30,30;40,50"/>
  </image>
</annotations>
"""


def test_cvat_import_parses_every_shape(tmp_path: Path):
    (tmp_path / "annotations.xml").write_text(CVAT_XML)
    frames = cvat_parse(tmp_path)
    assert len(frames) == 1
    f = frames[0]
    assert f.image_ref == "frame_000.jpg" and f.width == 640 and f.height == 480
    assert len(f.objects) == 4, [o.name for o in f.objects]

    box = next(o for o in f.objects if o.name == "autorickshaw")
    assert box.bbox == [10.0, 20.0, 110.0, 140.0]
    # a rotated box must not import as a wrong axis-aligned one
    assert box.rot_deg == 15.5
    assert box.attrs["overload"] == "true"

    poly = next(o for o in f.objects if o.name == "pedestrian")
    # points are a "x,y;x,y" STRING in CVAT; parsing them as floats naively yields nothing
    assert poly.mask_polygons == [[10.0, 10.0, 60.0, 10.0, 60.0, 80.0, 10.0, 80.0]]
    assert poly.bbox == [10.0, 10.0, 60.0, 80.0], "polygon bbox is the points AABB"

    line = next(o for o in f.objects if o.name == "curb")
    assert line.polyline == [[0.0, 470.0], [640.0, 460.0]]

    pts = next(o for o in f.objects if o.name == "pose")
    assert pts.keypoints["points"] == [[30.0, 30.0, 2], [40.0, 50.0, 2]]


def test_cvat_import_ignores_non_cvat_xml(tmp_path: Path):
    (tmp_path / "other.xml").write_text("<foo><bar/></foo>")
    with pytest.raises(FileNotFoundError):
        cvat_parse(tmp_path)


# ---- Label Studio ----------------------------------------------------------------------------------------

def _ls_task(x, y, w, h, ow=640, oh=480, label="sedan"):
    return {
        "id": 1,
        "data": {"image": "s3://bucket/frame_000.jpg"},
        "annotations": [{"result": [{
            "type": "rectanglelabels", "from_name": "label", "to_name": "image",
            "original_width": ow, "original_height": oh,
            "value": {"x": x, "y": y, "width": w, "height": h, "rectanglelabels": [label]},
        }]}],
    }


def test_labelstudio_percent_geometry_becomes_pixels(tmp_path: Path):
    """Label Studio stores rectangles in PERCENT. Reading them as pixels would pile every box in the
    top-left corner, which looks like a model failure rather than an import bug."""
    (tmp_path / "tasks.json").write_text(json.dumps([_ls_task(10, 25, 50, 50)]))
    frames = ls_parse(tmp_path)
    assert len(frames) == 1
    o = frames[0].objects[0]
    # 10% of 640 = 64, 25% of 480 = 120, +50% extents
    assert o.bbox == [64.0, 120.0, 384.0, 360.0], o.bbox
    assert o.name == "sedan"
    assert frames[0].width == 640 and frames[0].height == 480


def test_labelstudio_skips_geometry_without_original_size(tmp_path: Path):
    """Without original_width/height the percentages cannot be converted, so the shape is skipped rather
    than silently placed wrong."""
    task = _ls_task(10, 25, 50, 50)
    del task["annotations"][0]["result"][0]["original_width"]
    del task["annotations"][0]["result"][0]["original_height"]
    (tmp_path / "tasks.json").write_text(json.dumps([task]))
    assert ls_parse(tmp_path) == []


def test_labelstudio_excludes_predictions_by_default(tmp_path: Path):
    """Importing model predictions as though a human made them would poison the gold pool."""
    task = _ls_task(10, 10, 10, 10)
    task["predictions"] = [{"model_version": "m1", "result": [{
        "type": "rectanglelabels", "from_name": "label", "to_name": "image",
        "original_width": 640, "original_height": 480,
        "value": {"x": 50, "y": 50, "width": 10, "height": 10, "rectanglelabels": ["truck"]},
    }]}]
    (tmp_path / "tasks.json").write_text(json.dumps([task]))

    default = ls_parse(tmp_path)
    assert [o.name for o in default[0].objects] == ["sedan"], "predictions must not arrive as human labels"

    with_preds = ls_parse(tmp_path, include_predictions=True)
    assert sorted(o.name for o in with_preds[0].objects) == ["sedan", "truck"]


# ---- round trips -----------------------------------------------------------------------------------------

class _FakeStore:
    """Stands in for the object store: the export adapters only read mask blobs through get_bytes."""

    def __init__(self, blobs: dict[str, bytes] | None = None):
        self.blobs = blobs or {}

    def get_bytes(self, uri: str) -> bytes:
        return self.blobs[uri]


def _record(**kw):
    import uuid

    from services.export.records import ExportRecord

    base = dict(
        object_id=uuid.uuid4(), frame_id=uuid.uuid4(), session_id=uuid.uuid4(),
        ts_ns=1, cam_id="cam_f", img_uri="s3://b/f.jpg", width=640, height=480,
        vehicle_id="V1", city="BLR", class_id=1, class_name="sedan",
        bbox=[10.0, 20.0, 110.0, 140.0], conf=0.9, quality_score=None,
        state="accepted", source="human", mask_uri=None, mask_encoding=None,
        track_id=None, attrs={}, provenance={}, cuboid_3d=None, rot_deg=0.0,
        keypoints=None, polyline=None, relationships=[],
    )
    base.update(kw)
    return ExportRecord(**base)


def test_cvat_round_trip_preserves_box_and_rotation(tmp_path: Path):
    from services.autolabel.ontology import get_ontology
    from services.export.adapter_cvat import write_cvat

    onto = get_ontology()
    name = list(onto.classes)[0].name
    rec = _record(class_name=name, rot_deg=12.0, attrs={"overload": "true"})
    out = write_cvat([rec], onto, _FakeStore(), tmp_path / "cvat")

    # the file is valid CVAT and our own importer reads it back to the same numbers
    root = ET.parse(out).getroot()
    assert root.find("image") is not None
    back = cvat_parse(tmp_path / "cvat")
    assert len(back) == 1
    o = back[0].objects[0]
    assert o.bbox == [10.0, 20.0, 110.0, 140.0]
    assert o.rot_deg == 12.0, "rotation must survive the round trip"
    assert o.attrs["overload"] == "true"


def test_labelstudio_round_trip_preserves_pixel_geometry(tmp_path: Path):
    from services.autolabel.ontology import get_ontology
    from services.export.adapter_labelstudio import write_labelstudio

    onto = get_ontology()
    name = list(onto.classes)[0].name
    rec = _record(class_name=name, bbox=[64.0, 120.0, 384.0, 360.0])
    write_labelstudio([rec], onto, _FakeStore(), tmp_path / "ls")

    back = ls_parse(tmp_path / "ls")
    assert len(back) == 1
    o = back[0].objects[0]
    # pixels -> percent -> pixels must land back on the same box
    assert [round(v, 3) for v in o.bbox] == [64.0, 120.0, 384.0, 360.0], o.bbox
    assert o.name == name
