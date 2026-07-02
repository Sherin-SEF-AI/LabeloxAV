"""The Drift Investigator: turns a governance drift breach from an alert into a diagnosis.

When run_drift_scan reports a breach, this agent does the root-cause work the alert cannot: it pulls the
affected slice, localizes the breach to specific classes / scenes / sessions, looks for a common factor
(a vehicle, a city, a date cluster suggesting a firmware or pipeline change), forms a hypothesis, and
proposes an action (a narrow re-label, a kill-switch, or collect-more). It PROPOSES only -- it records an
AuditDecision and a report; a human disposes. Reuses run_drift_scan, the control sample, the label-drift
histogram, and disagreement mining; no model calls, so it is cheap to run on every breach.
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import ControlSample, Frame, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker

log = get_logger("agent.drift_investigator")

_KIND = "drift_investigator"


def _scene_of(frame: Frame) -> str:
    sc = frame.scene or {}
    return sc.get("weather") or sc.get("time_of_day") or "unknown"


async def _localize_control_precision(db: AsyncSession, metric: dict, onto) -> dict:
    """The auto-accept errors that dragged control precision down: which classes / scenes / sessions."""
    rows = (await db.execute(
        select(Object.class_id, Frame.scene, Frame.session_id)
        .select_from(ControlSample)
        .join(Object, Object.object_id == ControlSample.object_id)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(ControlSample.was_auto_accepted.is_(True), ControlSample.human_verdict == "incorrect"))).all()
    classes: Counter = Counter()
    scenes: Counter = Counter()
    sessions: Counter = Counter()
    for cid, scene, sid in rows:
        try:
            classes[onto.by_id(int(cid)).name] += 1
        except Exception:  # noqa: BLE001
            classes[str(cid)] += 1
        scenes[(scene or {}).get("weather") or (scene or {}).get("time_of_day") or "unknown"] += 1
        sessions[str(sid)] += 1
    common = await _common_factor(db, [s for s, _ in sessions.most_common(20)])
    return {"metric": "control_precision", "precision": metric.get("value"), "floor": metric.get("floor"),
            "n_errors": len(rows), "worst_classes": classes.most_common(5), "worst_scenes": scenes.most_common(3),
            "sessions": [s for s, _ in sessions.most_common(8)], "common_factor": common}


async def _localize_label_distribution(db: AsyncSession, drift: dict, onto) -> dict:
    """Which classes shifted most between the reference and current windows."""
    from services.govern.drift import _class_hist

    win = drift.get("window") or {}
    ref_ids, cur_ids = win.get("ref"), win.get("cur")
    n = 64
    ref = await _class_hist(db, ref_ids, n)
    cur = await _class_hist(db, cur_ids, n)
    rp = ref / max(1.0, ref.sum())
    cp = cur / max(1.0, cur.sum())
    movers = sorted(((i, float(cp[i] - rp[i])) for i in range(n)), key=lambda kv: abs(kv[1]), reverse=True)
    named = []
    for cid, delta in movers[:6]:
        if abs(delta) < 1e-4:
            continue
        try:
            named.append({"class": onto.by_id(cid).name, "delta_share": round(delta, 4)})
        except Exception:  # noqa: BLE001
            continue
    return {"metric": "label_distribution", "movers": named}


async def _common_factor(db: AsyncSession, session_ids: list[str]) -> dict | None:
    """A factor the affected sessions share: a vehicle, a city, or a tight capture-date cluster (which often
    means a hardware or pipeline change on that date)."""
    if not session_ids:
        return None
    rows = (await db.execute(select(DbSession.vehicle_id, DbSession.city, DbSession.start_ts_ns)
                             .where(DbSession.session_id.in_([uuid.UUID(s) for s in session_ids])))).all()
    if not rows:
        return None
    vehicles = Counter(v for v, _, _ in rows)
    cities = Counter(c for _, c, _ in rows if c)
    out: dict = {}
    if vehicles and vehicles.most_common(1)[0][1] >= max(2, len(rows) * 0.7):
        out["vehicle"] = vehicles.most_common(1)[0][0]
    if cities and cities.most_common(1)[0][1] >= max(2, len(rows) * 0.7):
        out["city"] = cities.most_common(1)[0][0]
    days = sorted(int(ts) // 86_400_000_000_000 for _, _, ts in rows if ts)
    if len(days) >= 2 and (days[-1] - days[0]) <= 3:
        out["date_cluster_days"] = days[-1] - days[0] + 1
    return out or None


def _synthesize(findings: list[dict]) -> tuple[str, dict]:
    """A one-line hypothesis and a proposed (never executed) action."""
    parts, action = [], {"kind": "collect_or_recheck", "detail": "gather more of the affected slice and re-scan"}
    for f in findings:
        if f["metric"] == "control_precision" and f.get("worst_classes"):
            cls = f["worst_classes"][0][0]
            loc = ""
            cf = f.get("common_factor") or {}
            if cf.get("vehicle"):
                loc = f", concentrated on vehicle {cf['vehicle']}"
            elif cf.get("date_cluster_days"):
                loc = f", in sessions clustered within {cf['date_cluster_days']} days (possible hardware/pipeline change)"
            elif f.get("worst_scenes"):
                loc = f", in {f['worst_scenes'][0][0]} scenes"
            parts.append(f"auto-accept precision dropped to {f.get('precision')} (floor {f.get('floor')}), "
                         f"driven by {cls} errors{loc}")
            action = {"kind": "narrow_relabel", "target_class": cls,
                      "sessions": f.get("sessions", []), "detail": f"re-label {cls} in the affected sessions and re-measure"}
        elif f["metric"] == "label_distribution" and f.get("movers"):
            m = f["movers"][0]
            parts.append(f"class mix shifted: {m['class']} share moved {m['delta_share']:+}")
        elif f["metric"] == "input_embedding":
            parts.append("input scene distribution shifted (novel imagery in the current window)")
    return ("; ".join(parts) or "breach detected but not localizable from available signals"), action


async def investigate(run_id: uuid.UUID, drift: dict) -> None:
    """Background worker: localize each breached metric, synthesize a hypothesis + proposed action, report."""
    from services.agent.runtime.report import finish_run
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    metrics = {m["metric"]: m for m in drift.get("metrics", [])}
    findings: list[dict] = []
    try:
        maker = get_sessionmaker()
        async with maker() as db:
            for name in drift.get("breached", []):
                if name == "control_precision":
                    findings.append(await _localize_control_precision(db, metrics.get(name, {}), onto))
                elif name == "label_distribution":
                    findings.append(await _localize_label_distribution(db, drift, onto))
                elif name == "input_embedding":
                    findings.append({"metric": "input_embedding", "note": "input distribution shifted"})
        hypothesis, action = _synthesize(findings)
        report = {"breached": drift.get("breached", []), "findings": findings,
                  "hypothesis": hypothesis, "proposed_action": action}
        await finish_run(run_id, status="committed", report=report, decision="root_cause_hypothesis")
        log.info("drift_investigate.done", run_id=str(run_id), breached=drift.get("breached"))
    except Exception as exc:  # noqa: BLE001
        log.error("drift_investigate.failed", run_id=str(run_id), error=str(exc))
        await finish_run(run_id, status="error", report={"error": str(exc)}, decision="root_cause_hypothesis")


async def launch_investigation(db: AsyncSession, drift: dict, *, created_by: str = _KIND) -> dict:
    from services.agent.runtime.report import launch

    async def _worker(run_id: uuid.UUID) -> None:
        await investigate(run_id, drift)

    return await launch(db, _KIND, _worker, created_by=created_by, policy={"breached": drift.get("breached", [])})


async def maybe_investigate(db: AsyncSession, drift: dict) -> dict:
    """On-breach hook for the runtime scheduler: investigate a breach at most once per hour (avoid a new
    investigation every controller tick while the breach persists)."""
    from datetime import datetime, timezone

    from services.agent.runtime.report import ran_since

    if not drift.get("breached"):
        return {"ran": False, "reason": "no breach"}
    if await ran_since(db, _KIND, datetime.now(timezone.utc) - timedelta(hours=1)):
        return {"ran": False, "reason": "already investigated recently"}
    res = await launch_investigation(db, drift, created_by="scheduler")
    return {"ran": True, **res}


async def run_on_demand(db: AsyncSession, ref_sessions=None, cur_sessions=None, *, created_by: str = _KIND) -> dict:
    """Scan for drift now and, if breached, launch an investigation. For the endpoint + tests."""
    from services.govern.drift import run_drift_scan

    drift = await run_drift_scan(db, ref_sessions, cur_sessions)
    if not drift.get("breached"):
        return {"breached": [], "ran": False}
    return {"breached": drift["breached"], **await launch_investigation(db, drift, created_by=created_by)}


async def latest_report(db: AsyncSession) -> dict | None:
    from services.agent.runtime.report import latest_run

    return await latest_run(db, _KIND)
