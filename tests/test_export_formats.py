"""Export format validation and the two adapters that closed the import/export round-trip holes.

The defect these guard: the export dispatcher was a chain of `if fmt in spec.formats` with no else, so a
requested format no adapter implemented was silently ignored. The job returned 200, wrote only the Parquet
sidecar, and recorded the never-written format on the DatasetCommit as if it had been delivered, i.e. we
shipped a dataset claiming contents it did not have.

Pure unit tests: the adapters are file writers over ExportRecord and the validator is a pure function, so
none of this needs a database."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from uuid import uuid4

import pytest

from services.autolabel.ontology import get_ontology
from services.export.adapter_mapillary import write_mapillary
from services.export.adapter_pascalvoc import write_pascalvoc
from services.export.dataset import (
    SUPPORTED_EXPORT_FORMATS,
    UnknownExportFormat,
    _requested,
    validate_formats,
)
from services.export.records import ExportRecord


def _rec(class_name: str = "pedestrian", bbox=None, width=640, height=480) -> ExportRecord:
    return ExportRecord(
        object_id=uuid4(), frame_id=uuid4(), session_id=uuid4(), ts_ns=1_700_000_000_000_000_000,
        cam_id="cam_f", img_uri="s3://bucket/frame.jpg", width=width, height=height,
        vehicle_id="TEST-01", city="BLR", class_id=get_ontology().by_name(class_name).id,
        class_name=class_name, bbox=bbox or [10.0, 20.0, 110.0, 220.0], conf=0.9,
        state="accepted", source="human",
    )


# ---- validation (the actual bug) ----

def test_unknown_format_is_refused_not_ignored():
    with pytest.raises(UnknownExportFormat) as exc:
        validate_formats(["coco", "waymo"])
    assert "waymo" in str(exc.value)          # names the offender
    assert "coco" in str(exc.value)           # and lists what is supported


def test_every_supported_format_validates():
    validate_formats(sorted(SUPPORTED_EXPORT_FORMATS))


def test_requested_drops_parquet_and_dedupes():
    # parquet is always written, so it is never dispatched; duplicates must not write twice.
    assert _requested(["coco", "parquet", "coco", "yolo"]) == ["coco", "yolo"]


def test_pascalvoc_and_mapillary_are_now_exportable():
    # Both were import-only, which made the round-trip lossy: a slice imported as VOC could not leave as VOC.
    assert {"pascalvoc", "mapillary"} <= SUPPORTED_EXPORT_FORMATS


# ---- Pascal VOC adapter ----

def test_pascalvoc_writes_parseable_xml_with_voc_indexing(tmp_path):
    onto = get_ontology()
    rec = _rec(bbox=[10.0, 20.0, 110.0, 220.0])
    out = write_pascalvoc([rec], onto, tmp_path / "voc")

    xmls = list((out / "Annotations").glob("*.xml"))
    assert len(xmls) == 1
    root = ET.parse(xmls[0]).getroot()
    assert root.findtext("size/width") == "640"
    obj = root.find("object")
    assert obj.findtext("name") == "pedestrian"
    # VOC is 1-indexed inclusive: xmin/ymin shift by +1, xmax/ymax stay (clamped into the image).
    assert obj.findtext("bndbox/xmin") == "11" and obj.findtext("bndbox/ymin") == "21"
    assert obj.findtext("bndbox/xmax") == "110" and obj.findtext("bndbox/ymax") == "220"
    assert (out / "ImageSets" / "Main" / "trainval.txt").read_text().strip() == xmls[0].stem


def test_pascalvoc_marks_edge_touching_box_truncated(tmp_path):
    onto = get_ontology()
    flush_left = _rec(bbox=[0.0, 20.0, 100.0, 220.0])
    out = write_pascalvoc([flush_left], onto, tmp_path / "voc")
    root = ET.parse(next((out / "Annotations").glob("*.xml"))).getroot()
    assert root.findtext("object/truncated") == "1"


def test_pascalvoc_clamps_box_inside_the_image(tmp_path):
    onto = get_ontology()
    overflow = _rec(bbox=[600.0, 400.0, 900.0, 700.0], width=640, height=480)
    out = write_pascalvoc([overflow], onto, tmp_path / "voc")
    root = ET.parse(next((out / "Annotations").glob("*.xml"))).getroot()
    assert int(root.findtext("object/bndbox/xmax")) <= 640
    assert int(root.findtext("object/bndbox/ymax")) <= 480


# ---- Mapillary adapter ----

def test_mapillary_maps_ontology_back_to_vistas_labels(tmp_path):
    onto = get_ontology()
    out = write_mapillary([_rec("pedestrian"), _rec("sedan")], onto, tmp_path / "map")
    payloads = [json.loads(p.read_text()) for p in (out / "polygons").glob("*.json")]
    labels = {o["label"] for pay in payloads for o in pay["objects"]}
    # the inverse of the importer's table, not our internal names
    assert "human--person--individual" in labels
    assert "object--vehicle--car" in labels


def test_distinct_frames_sharing_cam_and_timestamp_do_not_overwrite(tmp_path):
    # cam_id + ts_ns names a frame within one session, but a fleet export spans sessions and two vehicles can
    # hold the same camera name at the same nanosecond. Without a discriminator one frame's annotations
    # silently overwrote the other's, losing a whole frame from the export.
    onto = get_ontology()
    a, b = _rec("pedestrian"), _rec("sedan")          # different frame_id, identical cam_id and ts_ns
    assert a.frame_id != b.frame_id and (a.cam_id, a.ts_ns) == (b.cam_id, b.ts_ns)

    voc = write_pascalvoc([a, b], onto, tmp_path / "voc")
    assert len(list((voc / "Annotations").glob("*.xml"))) == 2

    mapi = write_mapillary([a, b], onto, tmp_path / "map")
    assert len(list((mapi / "polygons").glob("*.json"))) == 2


def test_mapillary_emits_box_as_closed_polygon(tmp_path):
    onto = get_ontology()
    out = write_mapillary([_rec(bbox=[1.0, 2.0, 3.0, 4.0])], onto, tmp_path / "map")
    payload = json.loads(next((out / "polygons").glob("*.json")).read_text())
    assert payload["objects"][0]["polygon"] == [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]]


def test_mapillary_records_unmapped_classes_instead_of_hiding_them(tmp_path):
    onto = get_ontology()
    # a class Vistas has no term for still exports, but the loss is recorded rather than silent
    out = write_mapillary([_rec("autorickshaw")], onto, tmp_path / "map")
    sidecar = json.loads((out / "unmapped.json").read_text())
    assert sidecar["counts"].get("autorickshaw") == 1
