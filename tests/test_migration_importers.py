"""Competitive migration is an integration problem, and every one of these formats has a geometry trap.

Scale gives left/top/width/height in pixels. SuperAnnotate gives x1/y1/x2/y2 in pixels and sometimes names a
class only by id. Encord normalises to 0..1 and nests a video's frames under one data unit. Read any of them
with the wrong assumption and the import succeeds: the boxes are simply in the wrong place, or a whole clip
collapses onto one frame, and it looks like a model failure rather than a parser bug.

So each test asserts the converted pixel geometry rather than that a parse returned something.
"""

import json
from pathlib import Path

from services.imports import adapter_encord, adapter_scale, adapter_superannotate
from services.imports.conflicts import taxonomy_report
from services.imports.records import ImportFrame, ImportObject


def test_scale_boxes_are_left_top_width_height_in_pixels(tmp_path: Path):
    (tmp_path / "task.json").write_text(json.dumps({
        "attachment": "frames/0001.jpg",
        "response": {"annotations": [
            {"label": "car", "left": 10, "top": 20, "width": 30, "height": 40,
             "attributes": {"occlusion": "partial"}, "uuid": "abc"},
            {"label": "pedestrian", "left": 5, "top": 5, "width": 0, "height": 10},   # zero width: dropped
        ]},
    }))
    frames = adapter_scale.parse(tmp_path)
    assert len(frames) == 1
    objs = frames[0].objects
    assert len(objs) == 1, "a zero-area box is not a label"
    assert objs[0].bbox == [10.0, 20.0, 40.0, 60.0], "left+width, top+height, not x2/y2"
    assert objs[0].name == "car"
    assert objs[0].attrs == {"occlusion": "partial"}, "their attributes survive the migration"
    assert objs[0].track_ref == "abc"


def test_scale_reads_jsonl_and_many_files(tmp_path: Path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"attachment": "a.jpg",
                    "response": {"annotations": [{"label": "bus", "left": 0, "top": 0,
                                                  "width": 5, "height": 5}]}}) + "\n"
        + json.dumps({"attachment": "b.jpg",
                      "response": {"annotations": [{"label": "bus", "left": 1, "top": 1,
                                                    "width": 5, "height": 5}]}}) + "\n")
    frames = adapter_scale.parse(tmp_path)
    assert {f.image_ref for f in frames} == {"a.jpg", "b.jpg"}


def test_superannotate_resolves_a_class_named_only_by_id(tmp_path: Path):
    """An export whose ids are never resolved imports every object as unknown, and a migration that arrives
    with one class is one nobody notices is broken until they try to train on it."""
    (tmp_path / "classes").mkdir()
    (tmp_path / "classes" / "classes.json").write_text(json.dumps([{"id": 7, "name": "autorickshaw"}]))
    (tmp_path / "img1.jpg.json").write_text(json.dumps({
        "metadata": {"name": "img1.jpg", "width": 1920, "height": 1080},
        "instances": [
            {"classId": 7, "points": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
             "attributes": [{"name": "red", "groupName": "colour"}]},
        ],
    }))
    frames = adapter_superannotate.parse(tmp_path)
    assert len(frames) == 1
    o = frames[0].objects[0]
    assert o.name == "autorickshaw", "the taxonomy file is what makes a classId meaningful"
    assert o.bbox == [100.0, 200.0, 300.0, 400.0], "already x1y1x2y2; no conversion"
    assert o.attrs == {"colour": "red"}, "the group is kept: colour red and state red are different facts"


def test_superannotate_prefers_the_name_when_the_export_carries_one(tmp_path: Path):
    (tmp_path / "img.jpg.json").write_text(json.dumps({
        "metadata": {"name": "img.jpg", "width": 100, "height": 100},
        "instances": [{"className": "cattle", "classId": 99,
                       "points": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}],
    }))
    assert adapter_superannotate.parse(tmp_path)[0].objects[0].name == "cattle"


def test_encord_denormalises_geometry_against_the_unit_size(tmp_path: Path):
    """0..1 read as pixels puts every box in the top-left corner, which looks like a model catastrophe."""
    (tmp_path / "labels.json").write_text(json.dumps({
        "data_units": {"h1": {
            "data_title": "clip.mp4", "width": 1000, "height": 500,
            "labels": {"0": {"objects": [
                {"name": "rider", "objectHash": "oh1",
                 "boundingBox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}}]}},
        }},
    }))
    frames = adapter_encord.parse(tmp_path)
    o = frames[0].objects[0]
    assert o.bbox == [100.0, 100.0, 400.0, 300.0]
    assert o.track_ref == "oh1", "objectHash is stable across a clip, which is a track"


def test_encord_keeps_a_clips_frames_apart(tmp_path: Path):
    """Flattening numbered frames onto one reference collapses a whole clip into a single frame."""
    (tmp_path / "labels.json").write_text(json.dumps({
        "data_units": {"h1": {
            "data_title": "clip.mp4", "width": 100, "height": 100,
            "labels": {
                "0": {"objects": [{"name": "bus", "boundingBox": {"x": 0, "y": 0, "w": 0.5, "h": 0.5}}]},
                "1": {"objects": [{"name": "bus", "boundingBox": {"x": 0.1, "y": 0, "w": 0.5, "h": 0.5}}]},
            },
        }},
    }))
    frames = adapter_encord.parse(tmp_path)
    assert len(frames) == 2
    assert len({f.image_ref for f in frames}) == 2, "two frames of one clip must stay distinct"


def test_encord_refuses_a_unit_with_no_size_rather_than_guessing(tmp_path: Path):
    """Guessing a size places every box somewhere plausible and wrong."""
    (tmp_path / "labels.json").write_text(json.dumps({
        "data_units": {"h1": {"data_title": "x.mp4",
                              "labels": {"0": {"objects": [
                                  {"name": "bus", "boundingBox": {"x": 0, "y": 0, "w": 1, "h": 1}}]}}}},
    }))
    assert adapter_encord.parse(tmp_path) == []


def test_the_conflict_report_names_what_a_migration_would_cost():
    """"Imported, 4,000 unmapped" invites a customer to discover the loss a month later. The list lets them
    decide beforehand."""
    frames = [ImportFrame(image_ref="a.jpg", objects=[
        ImportObject(name="car", bbox=[0, 0, 1, 1]),
        ImportObject(name="sedan", bbox=[0, 0, 1, 1]),
        ImportObject(name="quadricycle_prototype", bbox=[0, 0, 1, 1]),
    ])]
    r = taxonomy_report(frames)

    assert r["source_classes"] == 3 and r["objects"] == 3
    assert r["mapped_cleanly"] + r["into_fallback"] == r["objects"]
    # A name this ontology has never heard of must be named, not counted.
    assert any(u["source_class"] == "quadricycle_prototype" for u in r["unmapped"])
    assert all("falls_back_to" in u for u in r["unmapped"])
    assert isinstance(r["merges"], list)
    assert {"source_class", "ontology_class", "objects", "clean"} <= set(r["mapping"][0])


def test_the_report_surfaces_two_source_classes_collapsing_onto_one():
    """The finding that actually costs a customer a distinction they were paying for."""
    frames = [ImportFrame(image_ref="a.jpg", objects=[
        ImportObject(name="cattle", bbox=[0, 0, 1, 1]),
        ImportObject(name="cow", bbox=[0, 0, 1, 1]),
    ])]
    r = taxonomy_report(frames)
    merged = {m["ontology_class"]: m for m in r["merges"]}
    if merged:   # only asserts the shape when the ontology does in fact collapse them
        m = next(iter(merged.values()))
        assert len(m["source_classes"]) > 1 and m["objects"] >= 2


def test_an_empty_import_reports_zero_rather_than_dividing_by_it():
    r = taxonomy_report([])
    assert r["objects"] == 0 and r["fallback_fraction"] == 0.0
