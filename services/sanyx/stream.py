"""SANYX streaming health (M10). The batch path scores a session after it lands; this scores it AS it streams
from CHRONYX, so a bad recording is flagged mid-drive rather than only post-hoc. It accumulates per-window
frame stats and camera timestamps and re-runs the existing pure checks over the accumulated data after each
window, keeping a running score and raising an early flag the first time the running decision hits quarantine.

Reuses services/sanyx/checks and score unchanged; the checks are already windowable, so nothing is
re-derived. StreamingHealth is a plain accumulator, so mid-stream flagging is deterministic and testable."""

from __future__ import annotations

import numpy as np

from core.config import SanyxSettings
from services.sanyx import checks as C
from services.sanyx.score import aggregate


def _pseudo_seq(ts: list[int]) -> list[int]:
    if len(ts) < 2:
        return list(range(len(ts)))
    med = float(np.median(np.diff(sorted(ts)))) or 1.0
    t0 = min(ts)
    return [int(round((t - t0) / med)) for t in sorted(ts)]


class StreamingHealth:
    """Incremental session health. feed() a window of (camera timestamps, per-frame exposure stats, sampled
    gray frames); it returns the running partial report and sets flagged/first_flag_window when the running
    decision first reaches quarantine."""

    def __init__(self, cfg: SanyxSettings):
        self.cfg = cfg
        self._cam_ts: dict[str, list[int]] = {}
        self._exposures: list[dict] = []
        self._grays: list = []
        self.window = 0
        self.flagged = False
        self.first_flag_window: int | None = None
        self.last: dict | None = None

    def feed(self, cam_ts: dict[str, list[int]] | None = None, exposures: list[dict] | None = None,
             gray_frames: list | None = None) -> dict:
        self.window += 1
        for cam, ts in (cam_ts or {}).items():
            self._cam_ts.setdefault(cam, []).extend(ts)
        if exposures:
            self._exposures.extend(exposures)
        if gray_frames:
            self._grays.extend(gray_frames)

        results = [C.check_dropped_frames(self.cfg, {c: _pseudo_seq(ts) for c, ts in self._cam_ts.items()})]
        if self._exposures:
            results.append(C.check_exposure(self.cfg, self._exposures))
        if self._grays:
            results.append(C.check_lens_contamination(self.cfg, self._grays[-16:]))

        report = aggregate(results, self.cfg)
        report["window"] = self.window
        report["streaming"] = True
        if not self.flagged and report["decision"] == "quarantine":
            self.flagged = True
            self.first_flag_window = self.window
        report["flagged"] = self.flagged
        report["first_flag_window"] = self.first_flag_window
        self.last = report
        return report
