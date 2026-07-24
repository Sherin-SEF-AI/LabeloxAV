"""CVAT XML import (the "CVAT for images 1.1" annotation format).

CVAT is the tool most annotation teams already have work sitting in, so this is the migration path in. The
format nests every shape under its <image>, and carries per-shape <attribute> children that map onto our
ImportObject.attrs.

Two details that are easy to get wrong and silently lose data:

- CVAT polygon/polyline points are a "x,y;x,y" STRING, not an attribute list. Parsing them naively as floats
  yields nothing and the shape is dropped without an error.
- A <box> may carry `rotation`, which CVAT stores in degrees about the box centre. We keep the unrotated AABB
  in bbox (matching how Object stores oriented boxes) and carry the angle in rot_deg, so a rotated box does
  not import as a wrong axis-aligned one.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from core.logging import get_logger
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_cvat")


def _attrs(el: ET.Element) -> dict:
    """<attribute name="x">v</attribute> children become a plain dict."""
    out = {}
    for a in el.findall("attribute"):
        name = a.get("name")
        if name:
            out[name] = (a.text or "").strip()
    return out


def _points(raw: str | None) -> list[list[float]]:
    """CVAT stores points as 'x1,y1;x2,y2;...'."""
    if not raw:
        return []
    pts = []
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        try:
            x, y = pair.split(",")
            pts.append([float(x), float(y)])
        except ValueError:
            continue
    return pts


def _bbox_of(points: list[list[float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _find_xml(root: Path) -> list[Path]:
    # CVAT exports annotations.xml at the root; accept any xml that actually looks like CVAT.
    out = []
    for p in sorted(root.rglob("*.xml")):
        try:
            head = p.read_text(errors="ignore")[:400]
        except OSError:
            continue
        if "<annotations" in head:
            out.append(p)
    return out


def parse(root: Path) -> list[ImportFrame]:
    xmls = _find_xml(root)
    if not xmls:
        raise FileNotFoundError("no CVAT <annotations> xml found under the dataset")

    frames: list[ImportFrame] = []
    for xml in xmls:
        try:
            tree = ET.parse(xml)
        except ET.ParseError as exc:
            log.warning("import_cvat.parse_failed", file=xml.name, error=str(exc))
            continue

        for image in tree.getroot().findall("image"):
            name = image.get("name")
            if not name:
                continue
            w = int(float(image.get("width") or 0)) or None
            h = int(float(image.get("height") or 0)) or None
            objs: list[ImportObject] = []

            for box in image.findall("box"):
                try:
                    x1, y1 = float(box.get("xtl")), float(box.get("ytl"))
                    x2, y2 = float(box.get("xbr")), float(box.get("ybr"))
                except (TypeError, ValueError):
                    continue
                objs.append(ImportObject(
                    name=box.get("label") or "object_fallback",
                    bbox=[x1, y1, x2, y2],
                    attrs=_attrs(box),
                    rot_deg=float(box.get("rotation") or 0.0),
                    track_ref=box.get("track_id"),
                ))

            for poly in image.findall("polygon"):
                pts = _points(poly.get("points"))
                if len(pts) < 3:
                    continue
                objs.append(ImportObject(
                    name=poly.get("label") or "object_fallback",
                    bbox=_bbox_of(pts),
                    attrs=_attrs(poly),
                    # flattened [x,y,x,y,...], the shape services.imports.run writes to a mask blob
                    mask_polygons=[[c for p in pts for c in p]],
                    track_ref=poly.get("track_id"),
                ))

            for line in image.findall("polyline"):
                pts = _points(line.get("points"))
                if len(pts) < 2:
                    continue
                objs.append(ImportObject(
                    name=line.get("label") or "object_fallback",
                    bbox=_bbox_of(pts),
                    attrs=_attrs(line),
                    polyline=pts,
                    track_ref=line.get("track_id"),
                ))

            pts_el = image.findall("points")
            for pel in pts_el:
                pts = _points(pel.get("points"))
                if not pts:
                    continue
                objs.append(ImportObject(
                    name=pel.get("label") or "object_fallback",
                    bbox=_bbox_of(pts),
                    attrs=_attrs(pel),
                    # CVAT points carry no visibility flag; 2 = visible, matching the COCO convention
                    keypoints={"skeleton": pel.get("label") or "points",
                               "points": [[p[0], p[1], 2] for p in pts]},
                ))

            if not objs:
                continue
            frames.append(ImportFrame(image_ref=name, width=w, height=h, objects=objs))

    log.info("import_cvat.parsed", frames=len(frames), files=len(xmls))
    return frames
