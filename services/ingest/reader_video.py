"""Video + timestamped sidecar reader. The 'now' format in the spec (MP4/MOV dashcam).

Frames inherit ts_ns derived from a session start and the native fps (Principle 02: frame
indices are derived, never primary). A sidecar (CSV or JSON) supplies GNSS and CAN ego-speed,
matched to each frame by nearest timestamp.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

import cv2

from core.timebase import seconds_to_ns
from services.ingest.types import RawFrame, SideChannelSample, nearest_sample


def _load_sidecar(path: Path) -> list[SideChannelSample]:
    samples: list[SideChannelSample] = []
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text())
    else:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))

    for r in rows:
        def _f(key: str) -> float | None:
            v = r.get(key)
            if v is None or v == "":
                return None
            return float(v)

        ts = r.get("ts_ns")
        if ts is None or ts == "":
            continue
        samples.append(
            SideChannelSample(
                ts_ns=int(ts), lat=_f("lat"), lon=_f("lon"), ego_speed=_f("ego_speed")
            )
        )
    samples.sort(key=lambda s: s.ts_ns)
    return samples


def read_video(
    video_path: str | Path,
    cam_id: str,
    start_ts_ns: int,
    target_fps: float,
    sidecar_path: str | Path | None = None,
) -> Iterator[RawFrame]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    if native_fps <= 0:
        native_fps = target_fps
    step = max(1, round(native_fps / target_fps))

    samples = _load_sidecar(Path(sidecar_path)) if sidecar_path else []
    # Tolerance: half the labeling frame interval.
    max_gap_ns = seconds_to_ns(0.5 / target_fps) if target_fps > 0 else seconds_to_ns(1.0)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def _emit(idx: int, frame):
        ts_ns = start_ts_ns + seconds_to_ns(idx / native_fps)
        s = nearest_sample(samples, ts_ns, max_gap_ns)
        return RawFrame(
            ts_ns=ts_ns, cam_id=cam_id, image_bgr=frame,
            lat=s.lat if s else None, lon=s.lon if s else None, ego_speed=s.ego_speed if s else None,
        )

    try:
        # Sparse sampling (e.g. 1-3 fps from 30 fps 4K): seek to each target index instead of
        # decoding every frame. Falls back to sequential when the step is small.
        if step >= 8 and total > 0:
            for idx in range(0, total, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                yield _emit(idx, frame)
        else:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    yield _emit(idx, frame)
                idx += 1
    finally:
        cap.release()
