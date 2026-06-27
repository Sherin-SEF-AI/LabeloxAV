"""Fetch perception model weights into the scratch dir. Requires the ml extra installed.

    uv pip install -e ".[ml]"
    uv run python scripts/download_models.py

YOLO26 and SAM 3.1 weights are pulled by Ultralytics on first use; this pre-fetches them so the
first labeling run does not stall. The Qwen3-VL weights are pulled by the chosen VLM backend.
"""

from __future__ import annotations

import sys

from core.config import get_settings
from core.logging import get_logger, setup_logging

log = get_logger("download_models")


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    scratch = settings.scratch_path()

    try:
        from ultralytics import SAM, YOLO, YOLOWorld  # noqa: F401
    except ImportError:
        log.error("ml_extra_missing", hint='install with: uv pip install -e ".[ml]"')
        sys.exit(1)

    from ultralytics import SAM, YOLO, YOLOWorld

    ov = settings.models.openvocab

    log.info("download.path_a", weights=settings.models.yolo.weights)
    YOLO(settings.models.yolo.weights)  # triggers download into the ultralytics cache

    log.info("download.path_b_detector", weights=ov.detector_weights)
    YOLOWorld(ov.detector_weights)

    log.info("download.path_b_segmenter", weights=ov.seg_weights)
    SAM(ov.seg_weights)

    log.info(
        "download.vlm_note",
        backend=settings.models.vlm.backend,
        model=settings.models.vlm.model,
        note="VLM weights are fetched by the backend on first call (transformers/vllm/ollama).",
    )
    log.info("download.done", scratch=str(scratch))


if __name__ == "__main__":
    main()
