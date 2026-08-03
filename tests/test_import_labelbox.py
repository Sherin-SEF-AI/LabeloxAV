"""Labelbox import: the one platform a migration could not answer for.

Labelbox is the largest commercial annotation platform and had no adapter, so a team with years of labels
there had to route through COCO and lose everything COCO cannot carry, which for Labelbox is most of the
interesting part.

The tests are shaped around the two ways this goes silently wrong: parsing only one of the two live export
generations, and dropping the nested classifications a customer paid to collect.
"""

from __future__ import annotations

import json

from services.imports.adapter_labelbox import parse

# A current-generation export. Everything nests under data_row and projects.<id>.labels[].annotations.
MODERN = {
    "data_row": {"id": "ckx1", "external_id": "frame_0001.jpg", "global_key": "gk-1",
                 "row_data": "https://storage.labelbox.com/frame_0001.jpg"},
    "media_attributes": {"width": 1920, "height": 1080, "mime_type": "image/jpeg"},
    "projects": {
        "proj_abc": {
            "labels": [{
                "label_kind": "Default",
                "annotations": {
                    "objects": [
                        {"feature_id": "feat-1", "name": "autorickshaw",
                         "annotation_kind": "ImageBoundingBox",
                         "bounding_box": {"top": 100.0, "left": 200.0, "height": 50.0, "width": 80.0},
                         "classifications": [
                             {"name": "occlusion",
                              "radio_answer": {"name": "partial", "classifications": []}},
                             {"name": "damage", "checklist_answers": [
                                 {"name": "dented"}, {"name": "repainted"}]},
                             {"name": "note", "text_answer": {"content": "overloaded"}},
                         ]},
                        {"feature_id": "feat-2", "name": "drivable",
                         "annotation_kind": "ImagePolygon",
                         "polygon": [{"x": 0.0, "y": 500.0}, {"x": 1920.0, "y": 500.0},
                                     {"x": 1920.0, "y": 1080.0}, {"x": 0.0, "y": 1080.0}]},
                        {"feature_id": "feat-3", "name": "lane_marking",
                         "annotation_kind": "ImagePolyline",
                         "line": [{"x": 10.0, "y": 900.0}, {"x": 400.0, "y": 700.0}]},
                        {"feature_id": "feat-4", "name": "pothole",
                         "annotation_kind": "ImagePoint", "point": {"x": 640.0, "y": 800.0}},
                        {"feature_id": "feat-5", "name": "road",
                         "annotation_kind": "ImageSegmentationMask",
                         "mask": {"url": "https://api.labelbox.com/masks/feat-5.png"}},
                    ],
                    "classifications": [
                        {"name": "time_of_day", "radio_answer": {"name": "night"}},
                    ],
                },
            }]
        }
    },
}

# The older flat export. Same data, a completely different shape.
LEGACY = {
    "ID": "ck_legacy_1",
    "External ID": "frame_0002.jpg",
    "Labeled Data": "https://storage.labelbox.com/frame_0002.jpg",
    "Label": {
        "objects": [
            {"featureId": "old-1", "title": "cattle",
             "bbox": {"top": 10.0, "left": 20.0, "height": 30.0, "width": 40.0},
             "classifications": [{"title": "posture", "answer": {"title": "standing"}}]},
            {"featureId": "old-2", "title": "median",
             "line": [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]},
        ],
        "classifications": [{"title": "weather", "answer": {"title": "rain"}}],
    },
}


def _write(tmp_path, name, doc):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return p


# --- both generations ---------------------------------------------------------------------------


def test_the_current_export_generation_parses(tmp_path):
    _write(tmp_path, "export.json", [MODERN])
    frames = parse(tmp_path)
    assert len(frames) == 1
    fr = frames[0]
    assert fr.image_ref == "frame_0001.jpg"
    assert (fr.width, fr.height) == (1920, 1080)
    assert len(fr.objects) == 5


def test_the_legacy_export_generation_parses(tmp_path):
    """The customers this adapter exists for are the ones with years of history, and their export is likely
    to be the old shape. Handling only the current one would fail on exactly them."""
    _write(tmp_path, "export.json", [LEGACY])
    frames = parse(tmp_path)
    assert len(frames) == 1
    assert frames[0].image_ref == "frame_0002.jpg"
    assert {o.name for o in frames[0].objects} == {"cattle", "median"}


def test_a_directory_holding_both_generations_loses_neither(tmp_path):
    """Detected per record rather than per file: a real export directory can hold both after a partial
    re-export, and picking one shape for the whole directory would drop the other silently."""
    _write(tmp_path, "new.json", [MODERN])
    _write(tmp_path, "old.json", [LEGACY])
    frames = parse(tmp_path)
    assert {f.image_ref for f in frames} == {"frame_0001.jpg", "frame_0002.jpg"}


def test_ndjson_is_read_line_by_line(tmp_path):
    """Labelbox's own export streams NDJSON, which is what a large project actually downloads."""
    p = tmp_path / "export.ndjson"
    p.write_text(json.dumps(MODERN) + "\n" + json.dumps(LEGACY) + "\n")
    frames = parse(tmp_path)
    assert len(frames) == 2


# --- geometry -----------------------------------------------------------------------------------


def test_boxes_convert_from_top_left_plus_extents_to_xyxy(tmp_path):
    _write(tmp_path, "e.json", [MODERN])
    box = next(o for o in parse(tmp_path)[0].objects if o.name == "autorickshaw")
    assert box.bbox == [200.0, 100.0, 280.0, 150.0]


def test_a_polygon_survives_as_a_polygon_not_just_its_box(tmp_path):
    """Flattening to a bounding box is the loss a migration through COCO already forces. The whole point of
    a native adapter is that it does not."""
    _write(tmp_path, "e.json", [MODERN])
    poly = next(o for o in parse(tmp_path)[0].objects if o.name == "drivable")
    assert poly.mask_polygons == [[0.0, 500.0, 1920.0, 500.0, 1920.0, 1080.0, 0.0, 1080.0]]
    assert poly.bbox == [0.0, 500.0, 1920.0, 1080.0]


def test_a_polyline_survives_as_a_polyline(tmp_path):
    _write(tmp_path, "e.json", [MODERN])
    line = next(o for o in parse(tmp_path)[0].objects if o.name == "lane_marking")
    assert line.polyline == [[10.0, 900.0], [400.0, 700.0]]


def test_a_point_gets_an_extent_rather_than_being_dropped(tmp_path):
    """A zero-area box is rejected by the ingest quality gate, so a point feature would vanish. The location
    is the information; widening it by a pixel keeps the feature and loses nothing real."""
    _write(tmp_path, "e.json", [MODERN])
    pt = next(o for o in parse(tmp_path)[0].objects if o.name == "pothole")
    assert pt.bbox == [640.0, 800.0, 641.0, 801.0]


def test_a_segmentation_mask_url_is_recorded_not_fetched(tmp_path):
    """A parse that pulls image data turns a local dataset read into an authenticated network call. The uri
    is what run.py already knows how to carry."""
    _write(tmp_path, "e.json", [MODERN])
    mask = next(o for o in parse(tmp_path)[0].objects if o.name == "road")
    assert mask.mask_uri == "https://api.labelbox.com/masks/feat-5.png"
    assert mask.mask_encoding == "png_url"


# --- the part a migration silently loses ----------------------------------------------------------


def test_nested_classifications_are_kept_flattened_rather_than_dropped(tmp_path):
    """An attribute this ontology has no field for is still evidence somebody paid for. Dropping it during a
    migration is how a team discovers the switch cost them data."""
    _write(tmp_path, "e.json", [MODERN])
    box = next(o for o in parse(tmp_path)[0].objects if o.name == "autorickshaw")
    assert box.attrs["occlusion"] == "partial"
    assert box.attrs["damage"] == ["dented", "repainted"]
    assert box.attrs["note"] == "overloaded"


def test_frame_level_classifications_reach_every_object(tmp_path):
    """ImportFrame carries no attribute map, so a scene attribute somebody labelled would otherwise be
    dropped entirely."""
    _write(tmp_path, "e.json", [MODERN])
    for obj in parse(tmp_path)[0].objects:
        assert obj.attrs.get("time_of_day") == "night"


def test_a_per_object_attribute_wins_over_a_frame_level_one_of_the_same_name(tmp_path):
    """The more specific label is the one the annotator attached to the thing itself."""
    doc = json.loads(json.dumps(MODERN))
    doc["projects"]["proj_abc"]["labels"][0]["annotations"]["classifications"] = [
        {"name": "occlusion", "radio_answer": {"name": "none"}}]
    _write(tmp_path, "e.json", [doc])
    box = next(o for o in parse(tmp_path)[0].objects if o.name == "autorickshaw")
    assert box.attrs["occlusion"] == "partial"


def test_the_legacy_answer_shape_is_read_too(tmp_path):
    _write(tmp_path, "e.json", [LEGACY])
    cow = next(o for o in parse(tmp_path)[0].objects if o.name == "cattle")
    assert cow.attrs["posture"] == "standing"
    assert cow.attrs["weather"] == "rain"


def test_class_names_come_through_verbatim(tmp_path):
    """Every adapter returns the external name and services/imports/remap.py maps in one place, so taxonomy
    conflicts surface in one report rather than being resolved differently by each parser."""
    _write(tmp_path, "e.json", [MODERN])
    assert "autorickshaw" in {o.name for o in parse(tmp_path)[0].objects}


def test_feature_id_carries_identity_across_frames(tmp_path):
    """Labelbox keeps feature_id stable for a tracked feature, which is what track_ref means here."""
    _write(tmp_path, "e.json", [MODERN])
    box = next(o for o in parse(tmp_path)[0].objects if o.name == "autorickshaw")
    assert box.track_ref == "feat-1"


# --- refusing to invent ---------------------------------------------------------------------------


def test_a_record_with_no_image_reference_is_skipped_not_guessed(tmp_path):
    _write(tmp_path, "e.json", [{"data_row": {}, "projects": {}}])
    assert parse(tmp_path) == []


def test_a_degenerate_box_is_skipped(tmp_path):
    doc = {"data_row": {"external_id": "f.jpg"},
           "projects": {"p": {"labels": [{"annotations": {"objects": [
               {"name": "x", "bounding_box": {"top": 1, "left": 1, "height": 0, "width": 10}}]}}]}}}
    _write(tmp_path, "e.json", [doc])
    assert parse(tmp_path)[0].objects == []


def test_malformed_json_does_not_abort_the_whole_import(tmp_path):
    """One corrupt file in a large export should cost that file, not the migration."""
    (tmp_path / "broken.json").write_text("{not json")
    _write(tmp_path, "good.json", [MODERN])
    assert len(parse(tmp_path)) == 1


def test_the_adapter_is_registered_so_the_ui_can_offer_it():
    from services.imports.run import ADAPTERS, ALL_FORMATS

    assert "labelbox" in ADAPTERS
    assert "labelbox" in ALL_FORMATS


def test_the_menu_offers_exactly_what_the_backend_registers():
    """The comment above IMPORT_FORMATS in web/lib/menus.ts claims it mirrors the backend registry, and
    nothing enforced it.

    That is the same drift the menu/palette design exists to prevent, one layer down: a format listed in the
    UI and absent from ADAPTERS gives the user a 400 from a menu entry, and one registered but unlisted is a
    capability nobody can reach. Adding this adapter is exactly the moment the two diverge.
    """
    import re
    from pathlib import Path

    from services.imports.run import ALL_FORMATS

    src = Path(__file__).resolve().parent.parent / "web" / "lib" / "menus.ts"
    block = re.search(r"const IMPORT_FORMATS[^=]*=\s*\[(.*?)\];", src.read_text(), re.S)
    assert block, "IMPORT_FORMATS not found; this guard needs updating with the file"
    listed = set(re.findall(r'\["([a-z0-9_]+)"', block.group(1)))

    assert listed - set(ALL_FORMATS) == set(), (
        f"the menu offers formats the backend does not register: {sorted(listed - set(ALL_FORMATS))}")
    assert set(ALL_FORMATS) - listed == set(), (
        f"the backend registers formats the menu never offers: {sorted(set(ALL_FORMATS) - listed)}")
