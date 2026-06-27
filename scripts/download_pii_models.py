"""Fetch/verify the Gate A PII detector weights into .scratch/models/pii/.

Face: OpenCV YuNet (~345 KB, required for the gate). Plate: a config-pointed Ultralytics YOLO weight
(optional, swappable) - if you have a license-plate model, drop it at the configured plate path; the
gate blurs faces regardless. Mirrors scripts/download_models.py.

    uv run python scripts/download_pii_models.py
"""

from __future__ import annotations

from pathlib import Path

import httpx

from core.config import get_settings
from core.logging import get_logger, setup_logging

log = get_logger("download_pii")

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("pii.weights_present", path=str(dest))
        return True
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        log.info("pii.downloaded", path=str(dest), bytes=dest.stat().st_size)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("pii.download_failed", url=url, error=str(exc))
        if dest.exists():
            dest.unlink()
        return False


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    cfg = settings.pii

    face = Path(cfg.face_weights)
    ok = _download(YUNET_URL, face)
    if not ok:
        log.error("pii.face_missing", hint=f"download YuNet manually to {face}")

    plate = Path(cfg.plate_weights)
    if plate.exists():
        log.info("pii.plate_present", path=str(plate))
    else:
        log.warning(
            "pii.plate_absent",
            path=str(plate),
            note="optional; drop a license-plate YOLO .pt here to also blur plates",
        )

    log.info("pii.done", face_ok=ok, plate_present=plate.exists())


if __name__ == "__main__":
    main()
