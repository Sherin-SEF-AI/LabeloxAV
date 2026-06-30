"""Milestone A: the canonical session timeline. Every modality (camera, IMU, GNSS, audio) resolves to one
monotonic ts_ns axis, so a camera frame, an IMU sample, a GNSS fix, and an audio window refer to the same
instant. The alignment is pure index math over per-modality timestamp arrays, so it is data-source-agnostic:
it works the moment the MCAP demux lands IMU and audio, and is unit-testable with no infra.

Two invariants the math enforces: a GNSS gap wider than max_gap_ns is a dropout reported as such, never
interpolated into a false fix (a small offset shifts a fast agent over a metre at road speed); and the sync
method is reported per session (hardware when the recorder shares a clock, interpolated otherwise) with an
accumulated-error estimate when interpolated, so the UI can surface the uncertainty.
"""

from __future__ import annotations

import bisect

from core.logging import get_logger

log = get_logger("timeline")


def nearest_index(sorted_ts: list[int], t: int) -> int | None:
    """Index of the timestamp in sorted_ts nearest to t. None when empty."""
    if not sorted_ts:
        return None
    i = bisect.bisect_left(sorted_ts, t)
    if i == 0:
        return 0
    if i >= len(sorted_ts):
        return len(sorted_ts) - 1
    return i if abs(sorted_ts[i] - t) < abs(sorted_ts[i - 1] - t) else i - 1


def audio_sample_index(t_ns: int, audio_start_ns: int, sample_rate: int) -> int | None:
    """The audio sample index at t relative to the audio stream start, or None when t precedes the stream."""
    if sample_rate <= 0 or t_ns < audio_start_ns:
        return None
    return int(round((t_ns - audio_start_ns) / 1e9 * sample_rate))


def align_frame(frame_ts: int, imu_ts: list[int], gnss_ts: list[int],
                audio_start_ns: int | None, audio_sr: int | None, max_gap_ns: int) -> dict:
    """Map a frame timestamp to the nearest IMU sample, the nearest GNSS fix (None + dropout flag when the
    gap exceeds max_gap_ns, never a false interpolated fix), and the audio sample index."""
    out: dict = {"frame_ts_ns": frame_ts}
    ii = nearest_index(imu_ts, frame_ts)
    out["imu_index"] = ii
    out["imu_dt_ns"] = (imu_ts[ii] - frame_ts) if ii is not None else None

    gi = nearest_index(gnss_ts, frame_ts)
    if gi is not None and abs(gnss_ts[gi] - frame_ts) <= max_gap_ns:
        out["gnss_index"], out["gnss_dropout"] = gi, False
    else:
        out["gnss_index"], out["gnss_dropout"] = None, True   # a dropout is silence, not a fabricated fix

    out["audio_sample"] = (audio_sample_index(frame_ts, audio_start_ns, audio_sr)
                           if (audio_sr and audio_start_ns is not None) else None)
    return out


def sync_report(method: str, frame_ts: list[int], ref_ts: list[int]) -> dict:
    """The session sync method and, when interpolated, the accumulated-error estimate: the median residual
    between each frame and its nearest reference timestamp (hardware sync shares a clock, so zero drift)."""
    if method == "hardware":
        return {"sync_method": "hardware", "accumulated_error_ns": 0}
    if not frame_ts or not ref_ts:
        return {"sync_method": "interpolated", "accumulated_error_ns": None}
    residuals = sorted(abs(ref_ts[nearest_index(ref_ts, t)] - t) for t in frame_ts)
    median = residuals[len(residuals) // 2]
    return {"sync_method": "interpolated", "accumulated_error_ns": int(median)}


async def session_timeline(session_id) -> dict:
    """The canonical timeline for a session: which modalities are present, the ts range, and the sync report.
    Reads the frame and GNSS timestamps that exist today; IMU and audio arrays fill in once the MCAP demux
    lands them, with no change to this contract."""
    from sqlalchemy import select

    from db.models import Frame
    from db.session import get_sessionmaker
    async with get_sessionmaker()() as db:
        frame_ts = [int(t) for t in (await db.execute(
            select(Frame.ts_ns).where(Frame.session_id == session_id).order_by(Frame.ts_ns))).scalars().all()]
        # GNSS presence is by timestamp (coordinates are read by the ego-state service)
        gnss_ts = [int(t) for t in (await db.execute(
            select(Frame.ts_ns).where(Frame.session_id == session_id, Frame.gnss.isnot(None))
            .order_by(Frame.ts_ns))).scalars().all()]
    modalities = {"camera": len(frame_ts), "gnss": len(gnss_ts), "imu": 0, "audio": 0}
    rep = sync_report("interpolated", frame_ts, gnss_ts) if gnss_ts else {"sync_method": "none",
                                                                          "accumulated_error_ns": None}
    return {"session_id": str(session_id),
            "ts_range_ns": [frame_ts[0], frame_ts[-1]] if frame_ts else None,
            "modalities": modalities, "sync": rep,
            "note": "imu and audio arrays land when the chronyx MCAP demux is wired; the alignment contract "
                    "is unchanged"}
