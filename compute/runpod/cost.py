"""Cost model and safety guards for the warm cloud-GPU session. Cost is treated as correctness: the
accruing cost is always computable for the meter, and the guards decide when a connected pod MUST be torn
down so it can never linger. Pure functions, no I/O, so they are trivially testable with controlled time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CostConfig:
    hourly_usd: float
    idle_seconds: int          # auto-terminate after this long with no job running
    max_session_seconds: int   # hard cap: auto-terminate after this regardless of activity

    @classmethod
    def from_settings(cls, cloud) -> CostConfig:
        return cls(
            hourly_usd=float(cloud.warm_hourly_usd),
            idle_seconds=int(cloud.warm_idle_minutes * 60),
            max_session_seconds=int(cloud.warm_max_session_hours * 3600),
        )


def est_cost(gpu_seconds: float, hourly_usd: float) -> float:
    """Accrued cost from GPU seconds at the known hourly rate."""
    return round(max(0.0, gpu_seconds) / 3600.0 * hourly_usd, 4)


def gpu_seconds(started_at: datetime | None, now: datetime) -> float:
    """Seconds the GPU has been up. Zero until the pod reports running (started_at set)."""
    if started_at is None:
        return 0.0
    return max(0.0, (now - started_at).total_seconds())


def idle_remaining(idle_since: datetime | None, now: datetime, cfg: CostConfig) -> int | None:
    """Seconds until idle auto-terminate, or None when a job is running (not idle)."""
    if idle_since is None:
        return None
    return max(0, cfg.idle_seconds - int((now - idle_since).total_seconds()))


def session_remaining(max_session_until: datetime | None, now: datetime) -> int | None:
    """Seconds until the hard max-session cap auto-terminate."""
    if max_session_until is None:
        return None
    return max(0, int((max_session_until - now).total_seconds()))


def guard_breach(now: datetime, started_at: datetime | None, idle_since: datetime | None,
                 max_session_until: datetime | None, cfg: CostConfig) -> str | None:
    """The reason a connected pod must be torn down right now, or None. Max-session is the hard cap and is
    checked first; idle applies only when no job is running (idle_since set)."""
    if max_session_until is not None and now >= max_session_until:
        return "max_session"
    if idle_since is not None and (now - idle_since).total_seconds() >= cfg.idle_seconds:
        return "idle"
    return None
