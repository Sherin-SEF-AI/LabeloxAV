"""SANYX rig-level predictive maintenance (M10). A single HealthReport judges one session; this watches a
component's health across a vehicle's sessions over time, so a module degrading toward failure (a camera whose
lens or exposure sub-score is monotonically falling, an IMU whose integrity is dropping) is flagged BEFORE it
crosses the quarantine threshold. A linear fit over the per-check sub-scores gives a slope and a projection of
how many more sessions until the component fails.

detect_trends() is pure over the per-session sub-score series so it is unit-testable; rig_trends() loads the
session_health history and persists SanyxRigAlert rows."""

from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import SanyxRigAlert, SessionHealth
from db.models import Session as DbSession

log = get_logger("sanyx.trends")

# a check maps to the physical component it reports on
COMPONENT = {
    "dropped_frames": "camera_link", "exposure": "camera_exposure", "lens": "camera_optics",
    "gps": "gnss", "imu": "imu", "time_sync": "timesync",
}
_FAIL_SCORE = 0.3   # a sub-score at or below this is a failing component


def detect_trends(series_by_check: dict[str, list[float]], min_sessions: int = 4,
                  slope_warn: float = -0.03) -> list[dict]:
    """Detect components trending toward failure. series_by_check maps a check name to its per-session score
    (oldest to newest). Flags a check whose score is falling (negative slope past slope_warn) and heading for
    the fail threshold, with a projected sessions-to-failure and a severity."""
    alerts = []
    for check, series in series_by_check.items():
        s = [float(v) for v in series if v is not None]
        if len(s) < min_sessions:
            continue
        idx = np.arange(len(s))
        slope, intercept = np.polyfit(idx, s, 1)
        latest = s[-1]
        if slope >= slope_warn:
            continue  # stable or improving
        # sessions until the fit crosses the fail score
        remaining = (_FAIL_SCORE - latest) / slope if slope < 0 else None
        if remaining is not None and remaining < 0:
            severity = "critical"       # already at or below failing
        elif remaining is not None and remaining <= 3:
            severity = "warn"
        else:
            severity = "watch"
        alerts.append({
            "component": COMPONENT.get(check, check), "metric": check, "trend": "falling",
            "severity": severity,
            "evidence": {"slope": round(float(slope), 4), "latest_score": round(latest, 3),
                         "n_sessions": len(s), "projected_sessions_to_fail":
                         round(float(remaining), 1) if remaining is not None and remaining >= 0 else 0},
        })
    return alerts


async def rig_trends(db: AsyncSession, vehicle_id: str, limit: int = 50, persist: bool = True) -> dict:
    """Compute and (optionally) persist the health trends for a vehicle's rig from its session history."""
    rows = (await db.execute(
        select(SessionHealth).join(DbSession, SessionHealth.session_id == DbSession.session_id)
        .where(DbSession.vehicle_id == vehicle_id, SessionHealth.indexer_version.like("sanyx%"),
               SessionHealth.indexer_version != "sanyx-stream")
        .order_by(SessionHealth.created_at.asc()).limit(limit))).scalars().all()

    # One health record per session: a re-index or an operator override writes a second SessionHealth row for
    # the same session, and counting both would double-weight that session in the trend. Keep the latest
    # (created_at is ascending, so the last write for a session wins).
    latest_by_session = {h.session_id: h for h in rows}
    rows = sorted(latest_by_session.values(), key=lambda h: h.created_at)

    series: dict[str, list[float]] = {}
    for h in rows:
        for c in (h.checks or []):
            if c.get("score") is not None and c.get("name"):
                series.setdefault(c["name"], []).append(float(c["score"]))

    alerts = detect_trends(series)
    if persist:
        for a in alerts:
            db.add(SanyxRigAlert(vehicle_id=vehicle_id, component=a["component"], metric=a["metric"],
                                 trend=a["trend"], severity=a["severity"], evidence=a["evidence"]))
        if alerts:
            await db.commit()
    log.info("sanyx.rig_trends", vehicle=vehicle_id, sessions=len(rows), alerts=len(alerts))
    return {"vehicle_id": vehicle_id, "n_sessions": len(rows), "alerts": alerts,
            "series": {k: [round(v, 3) for v in vs] for k, vs in series.items()}}
