"""Milestone C: the timestamp-seam check. An event whose nearest camera frame is beyond the skew tolerance
(or has no frame at all) is flagged as a seam defect; an event aligned to a frame is not."""

from __future__ import annotations

from services.intelligence.consistency_qa import timestamp_seam_flags

SEC = 1_000_000_000
MS = 1_000_000


def test_aligned_event_is_not_flagged():
    frames = [0, SEC, 2 * SEC]
    events = [{"event_id": "a", "kind": "hard_brake", "modality": "imu", "t_start_ns": SEC + 10 * MS}]
    assert timestamp_seam_flags(events, frames, max_skew_ns=50 * MS) == []


def test_misaligned_event_is_a_seam():
    frames = [0, SEC, 2 * SEC]
    events = [{"event_id": "b", "kind": "horn", "modality": "audio", "t_start_ns": SEC + 300 * MS}]
    flags = timestamp_seam_flags(events, frames, max_skew_ns=50 * MS)
    assert len(flags) == 1 and flags[0]["reason"] == "timestamp_seam" and flags[0]["skew_ns"] == 300 * MS


def test_event_with_no_frames_has_no_visual_anchor():
    flags = timestamp_seam_flags([{"event_id": "c", "kind": "impact", "modality": "crossmodal",
                                   "t_start_ns": SEC}], frame_ts=[], max_skew_ns=SEC)
    assert len(flags) == 1 and flags[0]["reason"] == "no_visual_anchor" and flags[0]["skew_ns"] is None


def test_seams_sorted_worst_first():
    frames = [0]
    events = [
        {"event_id": "small", "kind": "k", "modality": "imu", "t_start_ns": 100 * MS},
        {"event_id": "big", "kind": "k", "modality": "imu", "t_start_ns": 900 * MS},
    ]
    flags = timestamp_seam_flags(events, frames, max_skew_ns=10 * MS)
    assert [f["event_id"] for f in flags] == ["big", "small"]   # larger skew first
