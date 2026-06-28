"""Fetch/verify the Gate A PII detector weights into .scratch/models/pii/.

Face: OpenCV YuNet (~345 KB, required for the gate). Plate: a config-pointed Ultralytics YOLO weight
(optional, swappable) - if you have a license-plate model, drop it at the configured plate path; the
gate blurs faces regardless. Mirrors scripts/download_models.py.

    uv run python scripts/download_pii_models.py
"""

from __future__ import annotations

import os
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
    headers = {}
    # Some mirrors (e.g. gated HuggingFace repos) need a token; honor one from the environment.
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {tok}"
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120, headers=headers) as r:
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
    plate_ok = plate.exists() and plate.stat().st_size > 0
    if plate_ok:
        log.info("pii.plate_present", path=str(plate))
    elif cfg.plate_url:
        plate_ok = _download(cfg.plate_url, plate)
        if not plate_ok:
            log.error(
                "pii.plate_download_failed",
                url=cfg.plate_url,
                hint=(f"set LBX_PII__PLATE_URL to a reachable Ultralytics .pt or drop one at {plate}; "
                      "with the gate on and plate_mandatory true, ingestion will fail until a plate "
                      "model is present (DPDPA: no silent plate leak)"),
            )
    else:
        log.error("pii.plate_absent", path=str(plate), hint="no plate_url configured; gate will fail loud")

    log.info("pii.done", face_ok=ok, plate_ok=plate_ok)


if __name__ == "__main__":
    main()
