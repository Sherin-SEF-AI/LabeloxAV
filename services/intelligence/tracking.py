"""Tracking: assign a stable track_id to per-frame objects (Plane 3 prerequisite). A deterministic
greedy-IoU tracker with a max-age tolerance, run per camera over frames in ts order.

ByteTrack (supervision) is the production upgrade for Kalman-smoothed association; the greedy
tracker is deterministic and keeps an exact object->track mapping, which the trajectory and event
stages depend on.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field

from core.config import get_settings
from services.autolabel.fusion import _iou


@dataclass
class Det:
    object_id: uuid.UUID
    frame_id: uuid.UUID
    ts_ns: int
    cam_id: str
    bbox: tuple[float, float, float, float]
    class_id: int
    embedding: object | None = None  # DINOv3 crop vector (L2-normalized np.ndarray) for re-ID, M2.0


@dataclass
class _ActiveTrack:
    track_id: uuid.UUID
    last_bbox: tuple[float, float, float, float]
    last_frame_index: int
    class_ids: Counter = field(default_factory=Counter)
    members: list[Det] = field(default_factory=list)


@dataclass
class TrackResult:
    track_id: uuid.UUID
    cam_id: str
    class_id: int
    first_ts_ns: int
    last_ts_ns: int
    members: list[Det]


def track_camera(dets: list[Det]) -> tuple[dict[uuid.UUID, uuid.UUID], list[TrackResult]]:
    """Track one camera's detections. Returns (object_id -> track_id, track summaries)."""
    cfg = get_settings().intelligence.tracker

    # Group detections by frame, preserving ts order.
    frames: dict[int, list[Det]] = {}
    for d in dets:
        frames.setdefault(d.ts_ns, []).append(d)
    ordered_ts = sorted(frames)

    active: list[_ActiveTrack] = []
    finished: list[_ActiveTrack] = []
    assignment: dict[uuid.UUID, uuid.UUID] = {}

    for fi, ts in enumerate(ordered_ts):
        # Retire tracks that have aged out.
        still: list[_ActiveTrack] = []
        for t in active:
            if fi - t.last_frame_index > cfg.max_age_frames:
                finished.append(t)
            else:
                still.append(t)
        active = still

        frame_dets = frames[ts]
        used: set[int] = set()
        # Greedy: highest-IoU pairs first.
        pairs = []
        for ti, t in enumerate(active):
            for di, d in enumerate(frame_dets):
                iou = _iou(t.last_bbox, d.bbox)
                if iou >= cfg.iou_match:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)
        matched_tracks: set[int] = set()
        for _iouv, ti, di in pairs:
            if ti in matched_tracks or di in used:
                continue
            t = active[ti]
            d = frame_dets[di]
            t.last_bbox = d.bbox
            t.last_frame_index = fi
            t.class_ids[d.class_id] += 1
            t.members.append(d)
            assignment[d.object_id] = t.track_id
            matched_tracks.add(ti)
            used.add(di)

        # Births for unmatched detections.
        for di, d in enumerate(frame_dets):
            if di in used:
                continue
            tid = uuid.uuid4()
            t = _ActiveTrack(track_id=tid, last_bbox=d.bbox, last_frame_index=fi)
            t.class_ids[d.class_id] += 1
            t.members.append(d)
            assignment[d.object_id] = tid
            active.append(t)

    finished.extend(active)

    results: list[TrackResult] = []
    keep_assignment: dict[uuid.UUID, uuid.UUID] = {}
    for t in finished:
        if len(t.members) < cfg.min_track_len:
            continue  # drop one-frame flickers; their objects stay untracked
        ms = sorted(t.members, key=lambda d: d.ts_ns)
        cid = t.class_ids.most_common(1)[0][0]
        results.append(
            TrackResult(
                track_id=t.track_id,
                cam_id=ms[0].cam_id,
                class_id=cid,
                first_ts_ns=ms[0].ts_ns,
                last_ts_ns=ms[-1].ts_ns,
                members=ms,
            )
        )
        for d in ms:
            keep_assignment[d.object_id] = t.track_id

    return keep_assignment, results
