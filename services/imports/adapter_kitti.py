"""KITTI import: inverts services.export.adapter_kitti. Reads each `<stem>.txt` label file (one
object per line: `type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y`) into an
ImportFrame, mapping `type` -> ImportObject.name and the four 2D bbox numbers -> xyxy pixel bbox.
The 3D fields are KITTI "unknown" sentinels in our 2D export and are ignored on the way back.

Each label matches its image by stem (the image dir is optional: KITTI labels are absolute-pixel, so
no image read is needed to reconstruct the box; the label stem itself names the frame).
"""

from __future__ import annotations

from pathlib import Path

from core.logging import get_logger
from services.imports.records import ImportFrame, ImportObject

log = get_logger("import_kitti")

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _find_image(label: Path, root: Path) -> Path | None:
    for ext in _IMG_EXT:
        hits = sorted(root.rglob(f"{label.stem}{ext}"))
        if hits:
            return hits[0]
    return None


def parse(root: Path) -> list[ImportFrame]:
    # KITTI labels live under a labels/ dir; fall back to any .txt that is not the classes vocabulary.
    label_files = sorted(root.rglob("*.txt"))
    label_files = [p for p in label_files if "labels" in p.parts or p.parent.name == "labels"] or [
        p for p in label_files if p.name != "classes.txt"
    ]
    if not label_files:
        raise FileNotFoundError("no KITTI label .txt files found under the dataset")

    frames: list[ImportFrame] = []
    for lf in label_files:
        objs: list[ImportObject] = []
        for line in lf.read_text().splitlines():
            parts = line.split()
            if len(parts) < 8:
                continue
            name = parts[0]
            x1, y1, x2, y2 = (float(v) for v in parts[4:8])
            objs.append(ImportObject(name=name, bbox=[x1, y1, x2, y2]))
        img = _find_image(lf, root)
        frames.append(ImportFrame(image_ref=str(img) if img else lf.stem, objects=objs))
    log.info("import_kitti.parsed", frames=len(frames))
    return frames
