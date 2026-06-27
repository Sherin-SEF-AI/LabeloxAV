"""UTC nanosecond time base. Principle 02: every timestamp is int64 UTC nanoseconds.

Frame indices are derived, never primary. Float seconds and local time are not used anywhere.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

NS_PER_SECOND = 1_000_000_000


def now_ns() -> int:
    """Current wall-clock time as int64 UTC nanoseconds."""
    return time.time_ns()


def datetime_to_ns(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp() * NS_PER_SECOND)


def ns_to_datetime(ts_ns: int) -> datetime:
    return datetime.fromtimestamp(ts_ns / NS_PER_SECOND, tz=timezone.utc)


def ns_to_iso(ts_ns: int) -> str:
    return ns_to_datetime(ts_ns).isoformat()


def seconds_to_ns(seconds: float) -> int:
    return int(round(seconds * NS_PER_SECOND))


def frame_ts_ns(t_start_ns: int, frame_index: int, fps: float) -> int:
    """Derive a frame timestamp from a session start, an index and a capture rate.

    Used only when a source has no per-frame stamp; an MCAP source stamps frames directly.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    return t_start_ns + seconds_to_ns(frame_index / fps)
