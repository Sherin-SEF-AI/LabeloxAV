"""SANYX-lite (M10): the on-device capture-QA subset that runs on the vehicle Jetson AGX Orin during a live
drive. It runs only the cheap CPU checks (exposure, dropped frames, lens) that fit the on-device budget, over
the streaming accumulator, and emits a compact operator status the driver/operator can read at a glance: a
green/amber/red state plus the worst active check. No GPU, no model, no object store: it reads the live camera
feed and reports.

Device-independent by construction (pure CPU + numpy + cv2), so it runs here against a replayed feed for the
acceptance and on the Jetson unchanged."""

from __future__ import annotations

import cv2
import numpy as np

from core.config import get_settings
from core.logging import get_logger
from services.ingest.quality import score_frame
from services.sanyx.stream import StreamingHealth

log = get_logger("sanyx.lite")

_STATE = {"pass": "OK", "degraded": "CHECK", "quarantine": "STOP"}


def _well_exposed(res) -> bool:
    return res.accepted or ("blur" not in " ".join(res.reasons) and res.clip_fraction < 0.2)


class SanyxLite:
    """On-device streaming QA. Push frames as they arrive; get back a compact operator status."""

    def __init__(self, cam_id: str = "cam_ft", window_frames: int = 15):
        self.cam_id = cam_id
        self.window_frames = window_frames
        self.sh = StreamingHealth(get_settings().sanyx)
        self._buf_ts: list[int] = []
        self._buf_exp: list[dict] = []
        self._buf_gray: list[np.ndarray] = []
        self.status: dict = {"state": "OK", "score": None, "worst": None}

    def push(self, frame_bgr: np.ndarray, ts_ns: int) -> dict:
        res = score_frame(frame_bgr, get_settings().ingest)
        self._buf_ts.append(int(ts_ns))
        self._buf_exp.append({"well_exposed": _well_exposed(res)})
        self._buf_gray.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        if len(self._buf_ts) >= self.window_frames:
            self._flush()
        return self.status

    def _flush(self) -> None:
        report = self.sh.feed({self.cam_id: self._buf_ts}, self._buf_exp, self._buf_gray)
        worst = min((c for c in report["checks"]), key=lambda c: c.get("score", 1.0), default=None)
        self.status = {
            "state": _STATE.get(report["decision"], "?"), "decision": report["decision"],
            "score": report["score"], "window": report["window"],
            "worst": worst["name"] if worst else None,
            "worst_detail": worst["detail"] if worst else None,
            "flagged": report["flagged"],
        }
        self._buf_ts, self._buf_exp, self._buf_gray = [], [], []
        log.info("sanyx.lite", cam=self.cam_id, state=self.status["state"], score=self.status["score"],
                 worst=self.status["worst"])
