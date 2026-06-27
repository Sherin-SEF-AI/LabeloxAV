"""Shared ingestion types. Readers yield RawFrame; the driver is reader-agnostic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RawFrame:
    ts_ns: int               # UTC nanoseconds, the frame identity
    cam_id: str
    image_bgr: np.ndarray
    lat: float | None = None
    lon: float | None = None
    ego_speed: float | None = None  # m/s, from CAN 0x247 when present


@dataclass
class SideChannelSample:
    ts_ns: int
    lat: float | None = None
    lon: float | None = None
    ego_speed: float | None = None


def nearest_sample(samples: list[SideChannelSample], ts_ns: int, max_gap_ns: int) -> SideChannelSample | None:
    """Nearest side-channel sample to a frame timestamp, within a tolerance."""
    if not samples:
        return None
    lo, hi = 0, len(samples) - 1
    # samples are sorted by ts_ns
    best = samples[0]
    best_gap = abs(samples[0].ts_ns - ts_ns)
    while lo <= hi:
        mid = (lo + hi) // 2
        gap = abs(samples[mid].ts_ns - ts_ns)
        if gap < best_gap:
            best_gap = gap
            best = samples[mid]
        if samples[mid].ts_ns < ts_ns:
            lo = mid + 1
        else:
            hi = mid - 1
    return best if best_gap <= max_gap_ns else None
