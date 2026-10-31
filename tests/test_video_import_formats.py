"""Every video container the upload page accepts must be one the importer recognises.

The page accepted .webm and .m4v while the importer stopped at .avi, so those uploads were stored and then
failed with "no video file found".
"""

from __future__ import annotations

import re
from pathlib import Path

from services.imports.run import VIDEO_EXTENSIONS

PAGE = Path(__file__).resolve().parent.parent / "web" / "app" / "annotate" / "new" / "page.tsx"


def test_the_importer_takes_every_video_the_upload_page_offers():
    accept = re.search(r'const ACCEPT = "([^"]+)"', PAGE.read_text()).group(1)
    offered = {e for e in accept.split(",") if e.startswith(".") and e not in (".zip", ".mcap")}
    assert offered <= set(VIDEO_EXTENSIONS), offered - set(VIDEO_EXTENSIONS)


def test_webm_decodes(tmp_path):
    import cv2
    import numpy as np
    import pytest

    out = tmp_path / "probe.webm"
    w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"VP80"), 5, (64, 48))
    if not w.isOpened():
        pytest.skip("this OpenCV build cannot write VP8; decoding is exercised by the sample import")
    for i in range(5):
        w.write(np.full((48, 64, 3), i * 40, np.uint8))
    w.release()
    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (48, 64, 3)
