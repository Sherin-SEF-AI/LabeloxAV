"""IDD-Detection (VOC XML) -> YOLO converter: offline unit test on a synthetic IDD tree."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.idd_to_yolo import convert


def _make_idd(root: Path):
    (root / "JPEGImages" / "scene").mkdir(parents=True)
    (root / "Annotations" / "scene").mkdir(parents=True)
    img = np.full((480, 640, 3), 120, dtype=np.uint8)
    cv2.imwrite(str(root / "JPEGImages" / "scene" / "0001.jpg"), img)
    xml = """<annotation>
  <size><width>640</width><height>480</height></size>
  <object><name>autorickshaw</name><bndbox><xmin>100</xmin><ymin>100</ymin><xmax>200</xmax><ymax>200</ymax></bndbox></object>
  <object><name>car</name><bndbox><xmin>300</xmin><ymin>200</ymin><xmax>400</xmax><ymax>280</ymax></bndbox></object>
  <object><name>train</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>60</xmax><ymax>60</ymax></bndbox></object>
</annotation>"""
    (root / "Annotations" / "scene" / "0001.xml").write_text(xml)
    (root / "train.txt").write_text("scene/0001\n")


def test_idd_to_yolo_conversion(tmp_path):
    idd = tmp_path / "IDD_Detection"
    _make_idd(idd)
    out = tmp_path / "idd_yolo"

    res = convert(idd, out, ["train"], link=False)
    assert res["images"] == 1
    # autorickshaw (kept), car->sedan, train->object_fallback
    names = res["ontology_classes"]
    assert "autorickshaw" in names and "sedan" in names and "object_fallback" in names

    data = yaml.safe_load((out / "data.yaml").read_text())
    name_to_idx = {v: k for k, v in data["names"].items()}

    label = next((out / "labels" / "train").glob("*.txt")).read_text().strip().splitlines()
    assert len(label) == 3  # all three objects remapped, none dropped

    # autorickshaw box (100,100,200,200) in 640x480 -> cx=0.234375 cy=0.3125 w=0.15625 h=0.208333
    auto_idx = name_to_idx["autorickshaw"]
    auto_line = next(l for l in label if int(l.split()[0]) == auto_idx)
    _, cx, cy, bw, bh = auto_line.split()
    assert abs(float(cx) - 0.234375) < 1e-4
    assert abs(float(cy) - 0.3125) < 1e-4
    assert abs(float(bw) - 0.15625) < 1e-4

    # indices are contiguous and ontology-id-ordered (autorickshaw 6 < sedan 11 < object_fallback 45)
    assert name_to_idx["autorickshaw"] < name_to_idx["sedan"] < name_to_idx["object_fallback"]
