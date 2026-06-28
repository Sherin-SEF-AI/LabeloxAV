"""BoT-SORT-style multi-object tracker (M2.0): Kalman constant-velocity motion + DINOv3 appearance,
associated by Hungarian assignment. The appearance feature is the Phase 1 DINOv3 object embedding on each
detection, so no separate re-ID model is loaded. Occlusion is handled by max-age; re-entry by appearance
re-acquisition; likely id-switches are flagged. Returns the same (object_id -> track_id, TrackResult[])
shape as the legacy greedy tracker so the miner integration is a drop-in, plus per-track id-switch flags.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.config import get_settings
from services.autolabel.fusion import _iou
from services.intelligence.tracking import Det, TrackResult

_INF = 1e3


def _cxcywh(b) -> np.ndarray:
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=np.float32)


def _xyxy(s) -> tuple[float, float, float, float]:
    cx, cy, w, h = float(s[0]), float(s[1]), float(s[2]), float(s[3])
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


class _KF:
    """Constant-velocity Kalman filter on [cx, cy, w, h] with velocity, state dim 8."""

    def __init__(self, bbox) -> None:
        self.x = np.zeros(8, np.float32)
        self.x[:4] = _cxcywh(bbox)
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 8), np.float32)
        for i in range(4):
            self.H[i, i] = 1.0
        self.Q = np.eye(8, dtype=np.float32) * 1.0
        self.R = np.eye(4, dtype=np.float32) * 10.0

    def predict(self) -> tuple:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _xyxy(self.x)

    def update(self, bbox) -> None:
        z = _cxcywh(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P


@dataclass
class _Track:
    track_id: uuid.UUID
    kf: _KF
    feat: np.ndarray | None
    last_frame_index: int
    class_ids: Counter = field(default_factory=Counter)
    members: list = field(default_factory=list)
    lost: int = 0
    pred: tuple = (0.0, 0.0, 0.0, 0.0)
    switches: list = field(default_factory=list)


def _spawn(d: Det, fi: int, tracks: list, assignment: dict) -> None:
    t = _Track(track_id=uuid.uuid4(), kf=_KF(d.bbox),
               feat=_norm(np.asarray(d.embedding, np.float32)) if d.embedding is not None else None,
               last_frame_index=fi)
    t.class_ids[d.class_id] += 1
    t.members.append(d)
    assignment[d.object_id] = t.track_id
    tracks.append(t)


def track_camera_botsort(dets: list[Det]) -> tuple[dict, list[TrackResult], dict]:
    """Track one camera's detections with BoT-SORT. Returns (object_id -> track_id, summaries,
    {track_id: [id_switch_flags]})."""
    cfg = get_settings().intelligence.tracker
    w_app = cfg.appearance_weight

    frames: dict[int, list[Det]] = {}
    for d in dets:
        frames.setdefault(d.ts_ns, []).append(d)
    ordered = sorted(frames)

    tracks: list[_Track] = []
    assignment: dict = {}

    for fi, ts in enumerate(ordered):
        fd = frames[ts]
        for t in tracks:
            t.pred = t.kf.predict()
        active = [t for t in tracks if t.lost <= cfg.max_age_frames]

        matched_d: set[int] = set()
        if active and fd:
            C = np.empty((len(active), len(fd)), np.float32)
            for i, t in enumerate(active):
                for j, d in enumerate(fd):
                    iou = _iou(t.pred, d.bbox)
                    cos = (float(t.feat @ _norm(np.asarray(d.embedding, np.float32)))
                           if (t.feat is not None and d.embedding is not None) else 0.0)
                    if iou >= cfg.iou_match or cos >= cfg.reid_cos:
                        C[i, j] = w_app * (1.0 - cos) + (1.0 - w_app) * (1.0 - iou)
                    else:
                        C[i, j] = _INF
            ri, ci = linear_sum_assignment(C)
            matched_t: set[int] = set()
            for i, j in zip(ri, ci, strict=False):
                if C[i, j] >= _INF:
                    continue
                t, d = active[i], fd[j]
                if t.lost > 0 and _iou(t.pred, d.bbox) < cfg.iou_match:  # re-acquired by appearance
                    t.switches.append({"frame_id": str(d.frame_id), "ts_ns": d.ts_ns, "reason": "reid_after_occlusion"})
                t.kf.update(d.bbox)
                if d.embedding is not None:
                    ev = _norm(np.asarray(d.embedding, np.float32))
                    t.feat = ev if t.feat is None else _norm(0.9 * t.feat + 0.1 * ev)
                t.class_ids[d.class_id] += 1
                t.members.append(d)
                t.last_frame_index = fi
                t.lost = 0
                assignment[d.object_id] = t.track_id
                matched_t.add(i)
                matched_d.add(j)
            for i, t in enumerate(active):
                if i not in matched_t:
                    t.lost += 1
        else:
            for t in tracks:
                t.lost += 1

        for j, d in enumerate(fd):
            if j not in matched_d:
                _spawn(d, fi, tracks, assignment)

    results: list[TrackResult] = []
    keep: dict = {}
    switches: dict = {}
    for t in tracks:
        if len(t.members) < cfg.min_track_len:
            continue
        ms = sorted(t.members, key=lambda d: d.ts_ns)
        cid = t.class_ids.most_common(1)[0][0]
        results.append(TrackResult(track_id=t.track_id, cam_id=ms[0].cam_id, class_id=cid,
                                    first_ts_ns=ms[0].ts_ns, last_ts_ns=ms[-1].ts_ns, members=ms))
        for d in ms:
            keep[d.object_id] = t.track_id
        if t.switches:
            switches[str(t.track_id)] = t.switches
    return keep, results, switches
