"""Fetch/verify the Gate A PII detector weights into .scratch/models/pii/.

Face: OpenCV YuNet (~227 KB, required whenever the gate is on). Plate: a config-pointed Ultralytics YOLO
weight, required whenever `pii.plate_mandatory` is set. Mirrors scripts/download_models.py.

    uv run python scripts/download_pii_models.py

Exits non-zero when a weight the gate would need is missing. It used to log the failure and exit 0, and on
2026-08-17 that cost a red build nobody could see coming: the `raw/main` YuNet URL 404'd for a two-hour
window, this script reported success, and the suite failed hours later inside ingest with
`PII gate enabled but the face detector is unavailable`. A provisioning step that cannot fail is not a
provisioning step, and the one it provisions here is the control that keeps unredacted faces out of the
object store. scripts/healthcheck.py already exits non-zero on the same condition; this now matches it.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

from core.config import get_settings
from core.logging import get_logger, setup_logging

log = get_logger("download_pii")

YUNET_URLS = (
    # Pinned to the commit that last touched this file rather than `raw/main`. A moving ref means an
    # upstream force-push, rename or transient outage silently changes what CI verifies; a commit SHA
    # cannot move under us. Verified 2026-08-18: 232,589 bytes.
    "https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    # Same bytes off the moving ref, tried only if the pinned object is ever unreachable.
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
)

_ATTEMPTS = 3
_BACKOFF_S = (1.0, 3.0)
# Below this a "download" is an error page or an LFS pointer, not a model. YuNet is ~227 KB and the
# smallest Ultralytics weight is ~5 MB, so anything under 8 KB is certainly not either.
_MIN_BYTES = 8 * 1024


def _fetch_once(url: str, dest: Path) -> bool:
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
        size = dest.stat().st_size
        if size < _MIN_BYTES:
            # A 200 carrying an error page or an unresolved git-lfs pointer. Treat it as a failure here
            # rather than letting the detector fail to load much later, somewhere less legible.
            log.error("pii.download_too_small", url=url, bytes=size, minimum=_MIN_BYTES)
            dest.unlink()
            return False
        log.info("pii.downloaded", path=str(dest), bytes=size)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("pii.download_attempt_failed", url=url, error=str(exc))
        if dest.exists():
            dest.unlink()
        return False


def _download(urls: str | Sequence[str], dest: Path) -> bool:
    """Fetch the first URL that yields a plausible weight, retrying each with backoff.

    The retry is not politeness, it is the fix for the observed failure: the outage that broke the build
    was transient, so a single attempt turned a blip into a red suite and a re-run would have hidden it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("pii.weights_present", path=str(dest))
        return True
    candidates = (urls,) if isinstance(urls, str) else tuple(urls)
    for url in candidates:
        for attempt in range(_ATTEMPTS):
            if _fetch_once(url, dest):
                return True
            if attempt < _ATTEMPTS - 1:
                time.sleep(_BACKOFF_S[attempt])
    log.error("pii.download_failed", urls=list(candidates), attempts=_ATTEMPTS)
    return False


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    cfg = settings.pii

    face = Path(cfg.face_weights)
    face_ok = _download(YUNET_URLS, face)
    if not face_ok:
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

    # The DB text detector behind the "text" redaction target: the plate the plate detector missed. Not
    # mandatory to fetch, because a deployment whose pack omits the target does not need it; but the
    # anonymizer refuses any frame when the target IS declared and the weights are absent, so a missing
    # one here is loud rather than silent.
    text_ok = False
    text_path = Path(cfg.text_weights) if cfg.text_weights else None
    if text_path is None:
        log.info("pii.text_not_configured",
                 hint="set LBX_PII__TEXT_WEIGHTS to enable the text redaction target")
    elif text_path.exists() and text_path.stat().st_size > 0:
        text_ok = True
        log.info("pii.text_present", path=str(text_path))
    elif cfg.text_url:
        text_ok = _download(cfg.text_url, text_path)
        if not text_ok:
            log.error("pii.text_download_failed", url=cfg.text_url,
                      hint=(f"drop an OpenCV DB text-detection ONNX at {text_path}; with the text target "
                            "declared and no weights, every frame is refused rather than passed"))

    log.info("pii.done", face_ok=face_ok, plate_ok=plate_ok, text_ok=text_ok)

    # Exit status mirrors exactly the condition under which the anonymizer refuses to construct
    # (services/anonymize/anonymizer.py), so a green step here means ingest will not fail on weights.
    missing = []
    if cfg.enabled and not face_ok:
        missing.append("face")
    if cfg.enabled and cfg.plate_mandatory and not plate_ok:
        missing.append("plate")
    if missing:
        log.error("pii.required_weights_missing", missing=missing,
                  hint="the DPDPA gate cannot be exercised without these; refusing to report success")
        sys.exit(1)


if __name__ == "__main__":
    main()
