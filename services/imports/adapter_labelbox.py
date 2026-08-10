"""Labelbox export import.

The largest commercial annotation platform with no path into this system, which made it the one question the
migration page could not answer. A team with years of labels in Labelbox had to re-export through COCO and
lose everything COCO cannot carry, which for Labelbox is most of the interesting part: nested
classifications, polylines, per-feature identity across frames.

**Two export generations, both live.** Labelbox changed its export shape, and which one a customer has
depends on when they last exported rather than on anything they chose. The modern one nests everything under
`data_row` and `projects.<id>.labels[].annotations`; the legacy one is a flat list with `External ID` and a
`Label` object. Handling only the current shape would fail on precisely the customers this exists for, the
ones with years of history, so both are parsed and the shape is detected per record rather than per file: a
real export directory can hold both after a partial re-export.

**Geometry.** `bounding_box` is `{top, left, height, width}` in pixels, the same top-left-plus-extents idiom
as Scale. Polygons and polylines arrive as `[{x, y}, ...]` rather than flat arrays. A segmentation mask
arrives as a URL to a PNG, which this adapter records rather than fetches: pulling image data during a parse
would turn a local dataset read into a network operation with credentials, and the mask uri is what
`services/imports/run.py` already knows how to carry.

**Classifications are theirs, and are kept.** Labelbox nests radio, checklist and free-text answers per
feature, arbitrarily deep. They are flattened into `attrs` under their own names rather than mapped, for the
same reason Scale's attributes are: an attribute this ontology has no field for is still evidence somebody
paid for, and dropping it during a migration is how a team discovers the switch cost them data. Class names
are likewise returned verbatim; `services/imports/remap.py` is the single place taxonomy is resolved.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.logging import get_logger
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_labelbox")


def _records(root: Path) -> list[dict]:
    """Every labelled data row in the export, whether it arrived as JSON, NDJSON, or many files."""
    out: list[dict] = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in (".json", ".ndjson", ".jsonl") or not p.is_file():
            continue
        text = p.read_text(errors="ignore").strip()
        if not text:
            continue
        if p.suffix.lower() in (".ndjson", ".jsonl"):
            for line in text.splitlines():
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            continue
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            continue
        out.extend(doc if isinstance(doc, list) else [doc])
    return out


def _flatten_classifications(items: list, prefix: str = "") -> dict:
    """Labelbox classifications, flattened to a name/value map.

    They nest arbitrarily: a radio answer can carry its own sub-classifications. Flattened with a dotted
    prefix rather than kept nested, because `attrs` is a flat map and a nested blob there would be
    unqueryable, which defeats the point of keeping them.
    """
    out: dict = {}
    for c in items or []:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or c.get("title") or c.get("value") or "").strip()
        if not name:
            continue
        key = f"{prefix}{name}"

        # Modern shape: one of these keys holds the answer.
        if "radio_answer" in c and isinstance(c["radio_answer"], dict):
            ans = c["radio_answer"]
            out[key] = ans.get("name") or ans.get("value")
            out.update(_flatten_classifications(ans.get("classifications"), f"{key}."))
        elif "checklist_answers" in c:
            answers = [a.get("name") or a.get("value") for a in (c["checklist_answers"] or [])
                       if isinstance(a, dict)]
            out[key] = [a for a in answers if a]
            for a in c["checklist_answers"] or []:
                if isinstance(a, dict):
                    out.update(_flatten_classifications(a.get("classifications"), f"{key}."))
        elif "text_answer" in c and isinstance(c["text_answer"], dict):
            out[key] = c["text_answer"].get("content")
        # Legacy shape: a single `answer`, which is a dict, a list, or a bare string.
        elif "answer" in c:
            ans = c["answer"]
            if isinstance(ans, dict):
                out[key] = ans.get("title") or ans.get("value")
            elif isinstance(ans, list):
                out[key] = [a.get("title") or a.get("value") for a in ans if isinstance(a, dict)]
            else:
                out[key] = ans
        if c.get("classifications"):
            out.update(_flatten_classifications(c["classifications"], f"{key}."))
    return {k: v for k, v in out.items() if v not in (None, "", [])}


def _points(raw) -> list[list[float]]:
    """`[{x, y}, ...]` to `[[x, y], ...]`. Anything unparseable is dropped rather than zero-filled."""
    pts: list[list[float]] = []
    for p in raw or []:
        if not isinstance(p, dict):
            continue
        try:
            pts.append([float(p["x"]), float(p["y"])])
        except (KeyError, TypeError, ValueError):
            continue
    return pts


def _bbox_of(pts: list[list[float]]) -> list[float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _bbox(box: dict) -> list[float] | None:
    """Labelbox boxes are top/left/height/width in pixels."""
    try:
        x, y = float(box["left"]), float(box["top"])
        w, h = float(box["width"]), float(box["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def _object(obj: dict) -> ImportObject | None:
    """One Labelbox feature, in whichever geometry it carries.

    The geometry key is what identifies the kind, not `annotation_kind`: the legacy export has no such field
    and the modern one is not always present on every feature. Checking the geometry works on both.
    """
    name = str(obj.get("name") or obj.get("title") or obj.get("value") or "object_fallback")
    attrs = _flatten_classifications(obj.get("classifications"))
    # feature_id is stable across frames for a tracked feature, which is what track_ref means here.
    track_ref = obj.get("feature_id") or obj.get("featureId") or obj.get("feature_schema_id")
    track_ref = str(track_ref) if track_ref else None

    box = obj.get("bounding_box") or obj.get("bbox")
    if isinstance(box, dict):
        bbox = _bbox(box)
        if bbox is None:
            return None
        return ImportObject(name=name, bbox=bbox, attrs=attrs, track_ref=track_ref)

    poly = _points(obj.get("polygon"))
    if len(poly) >= 3:
        return ImportObject(name=name, bbox=_bbox_of(poly), attrs=attrs, track_ref=track_ref,
                            mask_polygons=[[c for p in poly for c in p]])

    line = _points(obj.get("line"))
    if len(line) >= 2:
        return ImportObject(name=name, bbox=_bbox_of(line), attrs=attrs, track_ref=track_ref,
                            polyline=line)

    point = obj.get("point")
    if isinstance(point, dict):
        pts = _points([point])
        if pts:
            x, y = pts[0]
            # A point has no extent. Given a degenerate box the ingest quality gate would reject it, so it
            # is widened by a pixel: the location is the information, and losing the feature entirely to
            # preserve a zero-area box would be the worse trade.
            return ImportObject(name=name, bbox=[x, y, x + 1.0, y + 1.0], attrs=attrs, track_ref=track_ref)

    mask = obj.get("mask")
    if isinstance(mask, dict) and mask.get("url"):
        # Recorded, not fetched. A parse that pulls image data turns a local read into an authenticated
        # network operation, and the uri is what run.py already knows how to carry.
        return ImportObject(name=name, bbox=[0.0, 0.0, 0.0, 0.0], attrs=attrs, track_ref=track_ref,
                            mask_uri=str(mask["url"]), mask_encoding="png_url")

    return None


def _modern(rec: dict) -> tuple[str | None, int | None, int | None, list[dict], list]:
    """(image_ref, width, height, objects, frame_classifications) from a current-generation export."""
    dr = rec.get("data_row") or {}
    image_ref = dr.get("external_id") or dr.get("global_key") or dr.get("row_data") or dr.get("id")
    media = rec.get("media_attributes") or {}
    width, height = media.get("width"), media.get("height")

    objects: list[dict] = []
    frame_class: list = []
    for project in (rec.get("projects") or {}).values():
        if not isinstance(project, dict):
            continue
        for label in project.get("labels") or []:
            ann = (label or {}).get("annotations") or {}
            objects.extend(ann.get("objects") or [])
            frame_class.extend(ann.get("classifications") or [])
    return (str(image_ref) if image_ref else None, width, height, objects, frame_class)


def _legacy(rec: dict) -> tuple[str | None, int | None, int | None, list[dict], list]:
    """The same, from the older flat export."""
    image_ref = rec.get("External ID") or rec.get("Global Key") or rec.get("Labeled Data") or rec.get("ID")
    label = rec.get("Label") or {}
    if not isinstance(label, dict):
        return (None, None, None, [], [])
    return (str(image_ref) if image_ref else None, None, None,
            label.get("objects") or [], label.get("classifications") or [])


def parse(root: Path) -> list[ImportFrame]:
    frames: dict[str, ImportFrame] = {}
    skipped = 0
    seen_modern = seen_legacy = 0

    for rec in _records(root):
        if not isinstance(rec, dict):
            skipped += 1
            continue

        # Detected per record, not per file: a real export directory can hold both generations after a
        # partial re-export, and picking one shape for the whole directory would drop the other silently.
        if "data_row" in rec or "projects" in rec:
            image_ref, width, height, objects, frame_class = _modern(rec)
            seen_modern += 1
        elif "Label" in rec or "External ID" in rec:
            image_ref, width, height, objects, frame_class = _legacy(rec)
            seen_legacy += 1
        else:
            skipped += 1
            continue

        if not image_ref:
            skipped += 1
            continue

        fr = frames.get(image_ref)
        if fr is None:
            fr = ImportFrame(image_ref=image_ref, width=width, height=height)
            frames[image_ref] = fr
        elif fr.width is None and width:
            fr.width, fr.height = width, height

        # Frame-level classifications (weather, time of day) apply to every object on the frame. Attached to
        # each object rather than dropped, because ImportFrame carries no attribute map and a scene
        # attribute somebody labelled is exactly the kind of thing a migration must not lose.
        scene = _flatten_classifications(frame_class)

        for obj in objects:
            if not isinstance(obj, dict):
                skipped += 1
                continue
            parsed = _object(obj)
            if parsed is None:
                skipped += 1
                continue
            if scene:
                parsed.attrs = {**scene, **parsed.attrs}
            fr.objects.append(parsed)

    log.info("import.labelbox.parsed", frames=len(frames),
             objects=sum(len(f.objects) for f in frames.values()),
             modern_records=seen_modern, legacy_records=seen_legacy, skipped=skipped)
    return list(frames.values())
