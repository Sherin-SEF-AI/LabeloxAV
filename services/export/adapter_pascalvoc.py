"""Pascal VOC adapter: one XML annotation per frame, the format the VOC/ImageNet detection lineage uses and
that most legacy detection tooling still reads. Closes a round-trip hole: VOC was importable
(services/imports/adapter_pascalvoc.py) but not exportable, so a slice brought in as VOC could not be sent
back out in the format it arrived in.

VOC boxes are 1-indexed inclusive pixel corners, unlike our 0-indexed exclusive xyxy, so the conversion is
explicit below rather than a silent copy. Oriented boxes, masks, keypoints, and tracks have no VOC
representation and are dropped here by design; the Parquet sidecar remains the lossless record
(Principle 10: never block an export on format limits).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from services.autolabel.ontology import Ontology
from services.export.records import ExportRecord


def _stem(r: ExportRecord) -> str:
    """Stable image stem independent of img_uri availability, matching the KITTI adapter's convention."""
    return f"{r.cam_id}_{r.ts_ns}"


def unique_stems(by_frame: dict[str, list[ExportRecord]]) -> dict[str, str]:
    """Map frame_id -> a filename stem that is unique across the export.

    cam_id + ts_ns names a frame within one session, but a fleet export spans sessions and two vehicles can
    hold the same camera name at the same nanosecond. That collision would silently overwrite one frame's
    annotations with another's, so a colliding stem gets a short frame-id discriminator. Frames are processed
    in sorted frame-id order, making the assignment deterministic across runs.
    """
    stems: dict[str, str] = {}
    used: set[str] = set()
    for fkey in sorted(by_frame):
        base = _stem(by_frame[fkey][0])
        stem = base if base not in used else f"{base}_{fkey[:8]}"
        used.add(stem)
        stems[fkey] = stem
    return stems


def _truncated(r: ExportRecord) -> int:
    """VOC 'truncated' means the object extends past the image bounds. Derive it from the box touching an
    edge, which is what the flag records, rather than leaving it always 0."""
    x1, y1, x2, y2 = r.bbox
    return int(x1 <= 0 or y1 <= 0 or x2 >= r.width or y2 >= r.height)


def _difficult(r: ExportRecord) -> int:
    """VOC 'difficult' marks objects evaluation should ignore. Our closest honest signal is a heavy-occlusion
    attribute; absent that it is 0."""
    occ = r.attrs.get("occlusion")
    if isinstance(occ, bool):
        return int(occ)
    if isinstance(occ, int | float):
        return int(occ >= 75)
    return 0


def _annotation_xml(recs: list[ExportRecord], stem: str) -> ET.Element:
    first = recs[0]
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = "images"
    ET.SubElement(ann, "filename").text = f"{stem}.jpg"
    src = ET.SubElement(ann, "source")
    ET.SubElement(src, "database").text = "LabeloxAV"
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(first.width)
    ET.SubElement(size, "height").text = str(first.height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(ann, "segmented").text = "0"

    for r in recs:
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = r.class_name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(_truncated(r))
        ET.SubElement(obj, "difficult").text = str(_difficult(r))
        box = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = r.bbox
        # 0-indexed exclusive xyxy -> VOC 1-indexed inclusive, clamped inside the image.
        ET.SubElement(box, "xmin").text = str(max(1, int(round(x1)) + 1))
        ET.SubElement(box, "ymin").text = str(max(1, int(round(y1)) + 1))
        ET.SubElement(box, "xmax").text = str(min(r.width, int(round(x2))))
        ET.SubElement(box, "ymax").text = str(min(r.height, int(round(y2))))
    return ann


def write_pascalvoc(records: list[ExportRecord], onto: Ontology, out_dir: Path) -> Path:
    ann_dir = out_dir / "Annotations"
    sets_dir = out_dir / "ImageSets" / "Main"
    ann_dir.mkdir(parents=True, exist_ok=True)
    sets_dir.mkdir(parents=True, exist_ok=True)

    by_frame: dict[str, list[ExportRecord]] = {}
    for r in records:
        by_frame.setdefault(str(r.frame_id), []).append(r)

    stem_of = unique_stems(by_frame)
    for fkey, recs in by_frame.items():
        stem = stem_of[fkey]
        tree = ET.ElementTree(_annotation_xml(recs, stem))
        ET.indent(tree, space="  ")
        tree.write(ann_dir / f"{stem}.xml", encoding="utf-8", xml_declaration=True)

    # The VOC image set: the frame list an evaluator iterates. Sorted so the export is deterministic.
    stems = sorted(stem_of.values())
    (sets_dir / "trainval.txt").write_text("\n".join(stems) + "\n" if stems else "")
    ordered = sorted(onto.classes, key=lambda c: c.id)
    (out_dir / "classes.txt").write_text("\n".join(c.name for c in ordered) + "\n")
    return out_dir
