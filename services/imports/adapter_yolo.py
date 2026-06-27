"""YOLO import: inverts services.export.adapter_yolo. Reads data.yaml (class names) + labels/*.txt
(normalized cx cy w h) and pairs each label with its image. Boxes are denormalized to pixel xyxy
using the image dimensions.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import yaml

from core.logging import get_logger
from services.imports._util import find_file
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_yolo")

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _names(data_yaml: Path) -> dict[int, str]:
    doc = yaml.safe_load(data_yaml.read_text()) or {}
    raw = doc.get("names", {})
    if isinstance(raw, list):
        return {i: n for i, n in enumerate(raw)}
    return {int(k): v for k, v in raw.items()}


def _find_image(label: Path, images_dirs: list[Path]) -> Path | None:
    stem = label.stem
    for d in images_dirs:
        for ext in _IMG_EXT:
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    # last resort: search whole tree
    for ext in _IMG_EXT:
        hits = sorted(label.parents[1].rglob(f"{stem}{ext}"))
        if hits:
            return hits[0]
    return None


def parse(root: Path) -> list[ImportFrame]:
    data_yaml = find_file(root, "data.yaml", "*.yaml", "*.yml")
    if data_yaml is None:
        raise FileNotFoundError("no YOLO data.yaml found under the dataset")
    names = _names(data_yaml)

    label_files = sorted(root.rglob("*.txt"))
    label_files = [p for p in label_files if "labels" in p.parts or p.parent.name == "labels"] or label_files
    images_dirs = sorted({p.parent for p in root.rglob("*") if p.suffix.lower() in _IMG_EXT})

    frames: list[ImportFrame] = []
    for lf in label_files:
        img_path = _find_image(lf, images_dirs)
        if img_path is None:
            continue
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        objs: list[ImportObject] = []
        for line in lf.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            ci, cx, cy, bw, bh = int(float(parts[0])), *(float(x) for x in parts[1:5])
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            objs.append(ImportObject(name=names.get(ci, str(ci)), bbox=[x1, y1, x2, y2]))
        frames.append(ImportFrame(image_ref=str(img_path), width=w, height=h, objects=objs))
    log.info("import_yolo.parsed", frames=len(frames))
    return frames
