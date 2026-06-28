"""KITTI + BDD100K export/import adapters. Pure no-infra unit tests on the produced file structure,
plus a file round-trip oracle (export writer -> import parse) asserting object count + class names
are preserved. Modeled on test_p2_export_targets.py: ExportRecords are built directly and the writer
is called, no DB or MinIO needed (KITTI/BDD labels are text/json, no image fetch)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from services.autolabel.ontology import get_ontology
from services.export.adapter_bdd import write_bdd
from services.export.adapter_kitti import write_kitti
from services.export.records import ExportRecord
from services.imports.adapter_bdd import parse as parse_bdd
from services.imports.adapter_kitti import parse as parse_kitti


def _rec(object_id, frame_id, ts, class_id, class_name, bbox, track_id=None, attrs=None):
    return ExportRecord(
        object_id=object_id, frame_id=frame_id, session_id=uuid.UUID(int=1), ts_ns=ts, cam_id="cam_f",
        img_uri=f"s3://x/{frame_id}.jpg", width=640, height=480, vehicle_id="TIGOR-07", city="BLR",
        class_id=class_id, class_name=class_name, bbox=bbox, conf=0.9, state="auto_accept",
        source="auto_accept", track_id=track_id, attrs=attrs or {},
    )


def _two_frames():
    onto = get_ontology()
    f1, f2 = uuid.uuid4(), uuid.uuid4()
    a = _rec(uuid.uuid4(), f1, 1000, 6, "autorickshaw", [100, 100, 200, 200], attrs={"occlusion": 80})
    c = _rec(uuid.uuid4(), f1, 1000, 11, "sedan", [300, 300, 360, 360])
    b = _rec(uuid.uuid4(), f2, 2000, 6, "autorickshaw", [120, 100, 220, 200])
    return onto, [a, c, b]


# --- KITTI --------------------------------------------------------------------


def test_kitti_label_line_fields(tmp_path: Path):
    onto, recs = _two_frames()
    out = write_kitti(recs, onto, tmp_path / "kitti")

    # one label .txt per frame (two frames here) + a classes.txt vocabulary
    label_files = sorted((out / "labels").glob("*.txt"))
    assert len(label_files) == 2
    classes = (out / "classes.txt").read_text().splitlines()
    assert "autorickshaw" in classes and "sedan" in classes

    # the frame with two objects has two lines; check the 15-field KITTI 2D line shape + sentinels
    two = next(p for p in label_files if len(p.read_text().splitlines()) == 2)
    lines = two.read_text().splitlines()
    car_line = next(ln for ln in lines if ln.startswith("autorickshaw"))
    parts = car_line.split()
    assert len(parts) == 15
    assert parts[0] == "autorickshaw"
    assert parts[1] == "0.00"          # truncated
    assert parts[2] == "3"             # occluded (occlusion=80 -> 3)
    assert parts[3] == "-10.00"        # alpha sentinel
    assert [float(v) for v in parts[4:8]] == [100.0, 100.0, 200.0, 200.0]  # 2D bbox
    assert [float(v) for v in parts[8:11]] == [-1.0, -1.0, -1.0]           # h w l unknown
    assert [float(v) for v in parts[11:14]] == [-1000.0, -1000.0, -1000.0]  # x y z unknown
    assert float(parts[14]) == -10.0   # rotation_y unknown


def test_kitti_export_then_import_roundtrip(tmp_path: Path):
    onto, recs = _two_frames()
    out = write_kitti(recs, onto, tmp_path / "kitti")

    frames = parse_kitti(out)
    assert len(frames) == 2
    total = sum(len(f.objects) for f in frames)
    assert total == len(recs)  # object count preserved
    names = sorted(o.name for f in frames for o in f.objects)
    assert names == sorted(r.class_name for r in recs)  # class names preserved
    # the 2D bbox survives verbatim (KITTI is absolute pixel, no denormalization)
    boxes = {tuple(o.bbox) for f in frames for o in f.objects}
    assert (100.0, 100.0, 200.0, 200.0) in boxes


# --- BDD ----------------------------------------------------------------------


def test_bdd_json_shape(tmp_path: Path):
    onto, recs = _two_frames()
    path = write_bdd(recs, onto, tmp_path / "bdd")
    assert path.name == "bdd_det.json"

    doc = json.loads(path.read_text())
    assert isinstance(doc, list) and len(doc) == 2  # one entry per frame
    total = sum(len(e["labels"]) for e in doc)
    assert total == len(recs)

    entry = next(e for e in doc if len(e["labels"]) == 2)
    lab = entry["labels"][0]
    assert set(lab) >= {"id", "category", "box2d", "attributes"}
    assert set(lab["box2d"]) == {"x1", "y1", "x2", "y2"}
    assert lab["category"] in {"autorickshaw", "sedan"}
    # category names are the ontology class names; box matches a source record
    auto = next(o for e in doc for o in e["labels"] if o["category"] == "autorickshaw")
    assert auto["box2d"] == {"x1": 100, "y1": 100, "x2": 200, "y2": 200}


def test_bdd_export_then_import_roundtrip(tmp_path: Path):
    onto, recs = _two_frames()
    write_bdd(recs, onto, tmp_path / "bdd")

    frames = parse_bdd(tmp_path / "bdd")
    assert len(frames) == 2
    total = sum(len(f.objects) for f in frames)
    assert total == len(recs)  # object count preserved
    names = sorted(o.name for f in frames for o in f.objects)
    assert names == sorted(r.class_name for r in recs)  # class names preserved
    boxes = {tuple(o.bbox) for f in frames for o in f.objects}
    assert (100, 100, 200, 200) in boxes
