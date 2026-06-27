"""Pascal VOC import: reuses scripts.idd_to_yolo._parse_voc (the same parser the IDD converter uses).
Each Annotations/*.xml yields one ImportFrame; the image is found by matching stem under the dataset.
"""

from __future__ import annotations

from pathlib import Path

from core.logging import get_logger
from scripts.idd_to_yolo import _parse_voc
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_voc")

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _find_image(stem: str, root: Path) -> Path | None:
    for ext in _IMG_EXT:
        hits = sorted(root.rglob(f"{stem}{ext}"))
        if hits:
            return hits[0]
    return None


def parse(root: Path) -> list[ImportFrame]:
    xmls = sorted(root.rglob("*.xml"))
    if not xmls:
        raise FileNotFoundError("no Pascal VOC *.xml annotations found under the dataset")
    frames: list[ImportFrame] = []
    for xml in xmls:
        try:
            w, h, objs_raw = _parse_voc(xml)
        except Exception as exc:  # noqa: BLE001
            log.warning("import_voc.parse_failed", file=xml.name, error=str(exc))
            continue
        img = _find_image(xml.stem, root)
        if img is None:
            continue
        objs = [ImportObject(name=name, bbox=[x1, y1, x2, y2]) for name, x1, y1, x2, y2 in objs_raw]
        frames.append(ImportFrame(image_ref=str(img), width=w or None, height=h or None, objects=objs))
    log.info("import_voc.parsed", frames=len(frames))
    return frames
