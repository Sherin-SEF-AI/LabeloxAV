"""Raw-media import: video / mcap / image-folder. These reuse the ingest path verbatim so Gate A PII,
the quality gate, and the manifest all apply to imported footage exactly as to fleet captures. Only
read_image_folder is new (videos and mcaps already have readers).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2

from core.logging import get_logger
from core.timebase import now_ns, seconds_to_ns
from services.ingest.types import RawFrame

log = get_logger("import_raw")

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_image_folder(root: Path, fps: float = 3.0, cam_id: str = "cam_front") -> Iterator[RawFrame]:
    """Yield RawFrame for each image under root, synthesizing a monotonic timestamp at `fps` cadence
    so downstream frame identity (ts_ns) is unique and ordered."""
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXT)
    base = now_ns()
    step = seconds_to_ns(1.0 / max(fps, 0.1))
    for i, p in enumerate(images):
        im = cv2.imread(str(p))
        if im is None:
            continue
        yield RawFrame(ts_ns=base + i * step, cam_id=cam_id, image_bgr=im)
    log.info("import_raw.image_folder", count=len(images), root=str(root))
