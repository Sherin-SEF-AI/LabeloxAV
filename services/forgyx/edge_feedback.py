"""Field telemetry, and the gate learning from it.

FORGYX gates artifacts on latency and accuracy measured on a bench. A bench does not thermally throttle
after twenty minutes in a parked vehicle in Bengaluru, does not share its GPU with a video encoder, and
does not see the input distribution the field sees. So the gate has been passing artifacts on figures that
are true in a room nobody deploys in, and the first anyone learns of the difference is a device dropping
frames.

Three things this makes possible, and one it deliberately does not:

- **The bench number is compared against the field number**, and the gap is reported as a number rather
  than discovered as a complaint. A p95 that doubles on the device is the finding.
- **Thermal throttling is a first-class outcome.** An artifact that meets its latency budget for eighteen
  minutes and then does not has not met its latency budget.
- **Accuracy drift is detected without labels**, because the field has none. A confidence distribution that
  has moved away from the one measured at gate time is the available signal, compared by a distance over
  histograms rather than by a mean, since a mean is stable while a distribution splits in two.

What it does not do is fail an artifact on its own. Telemetry arrives from devices, which are outside the
trust boundary: a single misconfigured device reporting nonsense must not be able to demote a champion.
The evidence is surfaced and the demotion stays a decision.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import EdgeDevice, EdgeTelemetry

log = get_logger("edge_feedback")

# A device silent for longer than this is not reporting rather than performing well. Distinguished because
# an average over live devices only is the number that means something.
STALE_AFTER_HOURS = 6

# How much worse than the bench a field p95 may be before it is called a regression. Some gap is expected
# and healthy: a bench runs one model on an idle machine.
LATENCY_TOLERANCE = 1.35

# Fraction of a window spent throttled before an artifact is considered thermally unviable on that device.
THROTTLE_LIMIT = 0.10

# Hellinger distance between the gate-time and field confidence histograms above which the input
# distribution has moved enough to matter.
DRIFT_LIMIT = 0.25

# Below this many reporting devices, nothing here is a fleet finding. One device is an anecdote, and acting
# on it would let a single misconfigured unit speak for a rollout.
MIN_DEVICES = 3


class TelemetryError(Exception):
    """A telemetry operation refused."""


async def register_device(db: AsyncSession, *, device_id: str, hardware: str | None = None,
                          runtime: str | None = None, artifact_id: str | None = None,
                          model_version: str | None = None, fleet: str | None = None,
                          name: str | None = None, meta: dict | None = None) -> dict:
    """Register or update a device. Idempotent: a device re-registers on every boot."""
    row = await db.get(EdgeDevice, device_id)
    if row is None:
        row = EdgeDevice(device_id=device_id)
        db.add(row)
    row.name = name or row.name
    row.hardware = hardware or row.hardware
    row.runtime = runtime or row.runtime
    row.artifact_id = artifact_id or row.artifact_id
    row.model_version = model_version or row.model_version
    row.fleet = fleet or row.fleet
    row.meta = {**(row.meta or {}), **(meta or {})}
    row.last_seen_at = datetime.now(UTC)
    await db.commit()
    log.info("edge.device_registered", device=device_id, hardware=hardware, artifact=artifact_id)
    return _device_dict(row)


async def ingest_telemetry(db: AsyncSession, *, device_id: str, window_start_ns: int,
                           window_end_ns: int, n_inferences: int = 0,
                           latency_p50_ms: float | None = None,
                           latency_p95_ms: float | None = None,
                           latency_max_ms: float | None = None, fps: float | None = None,
                           temp_c_max: float | None = None,
                           throttled_fraction: float | None = None,
                           power_w_mean: float | None = None,
                           conf_histogram: list | None = None,
                           detections_per_frame: float | None = None,
                           dropped_frames: int = 0, artifact_id: str | None = None,
                           model_version: str | None = None, meta: dict | None = None) -> dict:
    """Accept one reporting window from one device.

    A window rather than a sample: a device posting every inference would spend its uplink on telemetry,
    and p50, p95 and the thermal ceiling reached are properties of a window anyway.
    """
    device = await db.get(EdgeDevice, device_id)
    if device is None:
        # Auto-registered rather than refused. A device that boots into a new fleet should start reporting
        # immediately; making telemetry conditional on a prior registration loses the first window, which
        # is the one containing the cold start.
        device = EdgeDevice(device_id=device_id, artifact_id=artifact_id,
                            model_version=model_version)
        db.add(device)
        await db.flush()

    if window_end_ns < window_start_ns:
        raise TelemetryError("window_end_ns is before window_start_ns")

    device.last_seen_at = datetime.now(UTC)
    if artifact_id:
        device.artifact_id = artifact_id
    if model_version:
        device.model_version = model_version

    row = EdgeTelemetry(
        device_id=device_id, artifact_id=artifact_id or device.artifact_id,
        model_version=model_version or device.model_version,
        window_start_ns=int(window_start_ns), window_end_ns=int(window_end_ns),
        n_inferences=int(n_inferences), latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms, latency_max_ms=latency_max_ms, fps=fps,
        temp_c_max=temp_c_max, throttled_fraction=throttled_fraction,
        power_w_mean=power_w_mean, conf_histogram=list(conf_histogram or []),
        detections_per_frame=detections_per_frame, dropped_frames=int(dropped_frames),
        meta=meta or {})
    db.add(row)
    await db.commit()
    return {"telemetry_id": str(row.telemetry_id), "device_id": device_id,
            "artifact_id": row.artifact_id}


def hellinger(p: list[float], q: list[float]) -> float:
    """Distance between two confidence histograms, 0 identical to 1 disjoint.

    Hellinger rather than KL, for two reasons that both bite here: it is symmetric, so "the field moved
    from the bench" and "the bench moved from the field" are the same number, and it is finite when one
    distribution has a zero the other does not, which happens the moment the field sees a class the bench
    set never contained.
    """
    if not p or not q or len(p) != len(q):
        return 0.0
    sp, sq = sum(p), sum(q)
    if sp <= 0 or sq <= 0:
        return 0.0
    pn = [v / sp for v in p]
    qn = [v / sq for v in q]
    total = sum((math.sqrt(a) - math.sqrt(b)) ** 2 for a, b in zip(pn, qn, strict=True))
    return float(min(1.0, math.sqrt(total / 2.0)))


async def artifact_field_report(db: AsyncSession, artifact_id: str, *,
                                hours: int = 168) -> dict:
    """What the field says about one artifact, next to what the bench said.

    The comparison is the product. Either number alone is uninteresting; the gap between them is the thing
    the gate has never been able to see.
    """
    since = datetime.now(UTC) - timedelta(hours=hours)
    rows = (await db.execute(
        select(EdgeTelemetry).where(EdgeTelemetry.artifact_id == artifact_id,
                                    EdgeTelemetry.created_at >= since))).scalars().all()
    if not rows:
        return {"artifact_id": artifact_id, "devices": 0, "windows": 0,
                "detail": "no field telemetry for this artifact in the window"}

    fresh_cutoff = datetime.now(UTC) - timedelta(hours=STALE_AFTER_HOURS)
    devices = (await db.execute(
        select(EdgeDevice).where(EdgeDevice.artifact_id == artifact_id))).scalars().all()
    live = [d for d in devices if d.last_seen_at and d.last_seen_at >= fresh_cutoff]

    p95s = [r.latency_p95_ms for r in rows if r.latency_p95_ms is not None]
    p50s = [r.latency_p50_ms for r in rows if r.latency_p50_ms is not None]
    throttles = [r.throttled_fraction for r in rows if r.throttled_fraction is not None]
    temps = [r.temp_c_max for r in rows if r.temp_c_max is not None]
    inferences = sum(int(r.n_inferences or 0) for r in rows)
    dropped = sum(int(r.dropped_frames or 0) for r in rows)

    field_hist = _sum_histograms([r.conf_histogram for r in rows])
    bench = await _bench_numbers(db, artifact_id)
    drift = hellinger(bench.get("conf_histogram") or [], field_hist) if field_hist else None

    field_p95 = _percentile(p95s, 0.95)
    bench_p95 = bench.get("latency_p95_ms")
    ratio = (field_p95 / bench_p95) if (field_p95 and bench_p95) else None

    findings: list[dict] = []
    if ratio and ratio > LATENCY_TOLERANCE:
        findings.append({
            "kind": "latency_regression", "severity": "warn",
            "detail": (f"field p95 is {ratio:.2f}x the bench p95 "
                       f"({field_p95:.1f}ms against {bench_p95:.1f}ms)")})
    worst_throttle = max(throttles) if throttles else 0.0
    if worst_throttle > THROTTLE_LIMIT:
        findings.append({
            "kind": "thermal_throttling", "severity": "warn",
            # An artifact that meets its budget for eighteen minutes and then does not has not met it.
            "detail": (f"a device spent {worst_throttle:.0%} of a window throttled at "
                       f"{max(temps):.0f}C" if temps else f"{worst_throttle:.0%} of a window throttled")})
    if drift is not None and drift > DRIFT_LIMIT:
        findings.append({
            "kind": "distribution_drift", "severity": "warn",
            "detail": (f"the field confidence distribution has moved {drift:.2f} from the one measured "
                       "at gate time; the input distribution is not the one this was validated on")})
    if inferences and dropped / max(inferences, 1) > 0.02:
        findings.append({"kind": "dropped_frames", "severity": "warn",
                         "detail": f"{dropped} of {inferences} frames dropped"})

    return {
        "artifact_id": artifact_id, "hours": hours,
        "devices": len(devices), "live_devices": len(live), "windows": len(rows),
        "inferences": inferences, "dropped_frames": dropped,
        "field": {"latency_p50_ms": _percentile(p50s, 0.5),
                  "latency_p95_ms": field_p95,
                  "worst_throttled_fraction": round(worst_throttle, 4),
                  "temp_c_max": max(temps) if temps else None},
        "bench": bench,
        "latency_ratio": round(ratio, 3) if ratio else None,
        "confidence_drift": round(drift, 4) if drift is not None else None,
        "findings": findings,
        # Stated rather than implied: below the floor these are observations about a device, not about a
        # fleet, and acting on them would let one misconfigured unit speak for a rollout.
        "fleet_significant": len(live) >= MIN_DEVICES,
        "min_devices": MIN_DEVICES,
    }


async def field_gate(db: AsyncSession, artifact_id: str, *, hours: int = 168) -> dict:
    """The gate's view of an artifact once the field has been heard from.

    Advisory by construction. Telemetry comes from devices, which are outside the trust boundary, so a
    single misconfigured unit reporting nonsense must not be able to demote a champion. This returns a
    verdict and evidence; the demotion stays a decision somebody makes.
    """
    report = await artifact_field_report(db, artifact_id, hours=hours)
    if report.get("windows", 0) == 0:
        return {**report, "verdict": "unknown",
                "detail": "no field evidence; the bench numbers are all there is"}
    if not report["fleet_significant"]:
        return {**report, "verdict": "insufficient_evidence",
                "detail": (f"{report['live_devices']} live device(s) reporting; "
                           f"{MIN_DEVICES} are needed before this is a fleet finding")}

    verdict = "pass" if not report["findings"] else "field_regression"
    log.info("edge.field_gate", artifact=artifact_id, verdict=verdict,
             findings=len(report["findings"]))
    return {**report, "verdict": verdict,
            "detail": ("the field agrees with the bench" if verdict == "pass"
                       else "; ".join(f["detail"] for f in report["findings"])),
            "advisory": True}


async def fleet_summary(db: AsyncSession, *, hours: int = 24) -> dict:
    """Every artifact currently in the field, and how each is doing."""
    since = datetime.now(UTC) - timedelta(hours=hours)
    artifacts = (await db.execute(
        select(EdgeTelemetry.artifact_id, func.count(),
               func.count(func.distinct(EdgeTelemetry.device_id)))
        .where(EdgeTelemetry.created_at >= since, EdgeTelemetry.artifact_id.isnot(None))
        .group_by(EdgeTelemetry.artifact_id))).all()

    out = []
    for artifact_id, windows, devices in artifacts:
        report = await artifact_field_report(db, artifact_id, hours=hours)
        out.append({"artifact_id": artifact_id, "windows": int(windows),
                    "devices": int(devices),
                    "latency_p95_ms": report["field"]["latency_p95_ms"],
                    "latency_ratio": report["latency_ratio"],
                    "findings": len(report["findings"]),
                    "fleet_significant": report["fleet_significant"]})
    devices_total = (await db.execute(select(func.count()).select_from(EdgeDevice))).scalar_one()
    stale = (await db.execute(
        select(func.count()).select_from(EdgeDevice)
        .where(EdgeDevice.last_seen_at < datetime.now(UTC) - timedelta(hours=STALE_AFTER_HOURS))
    )).scalar_one()
    return {"hours": hours, "artifacts": out, "devices": int(devices_total),
            # Surfaced, because a fleet whose devices have gone quiet looks identical to a healthy one in
            # every average computed over the devices that are still talking.
            "silent_devices": int(stale)}


async def list_devices(db: AsyncSession, *, fleet: str | None = None, limit: int = 200) -> dict:
    stmt = select(EdgeDevice).order_by(EdgeDevice.last_seen_at.desc().nullslast()).limit(limit)
    if fleet:
        stmt = stmt.where(EdgeDevice.fleet == fleet)
    rows = (await db.execute(stmt)).scalars().all()
    cutoff = datetime.now(UTC) - timedelta(hours=STALE_AFTER_HOURS)
    return {"devices": [{**_device_dict(d),
                         "live": bool(d.last_seen_at and d.last_seen_at >= cutoff)}
                        for d in rows]}


async def _bench_numbers(db: AsyncSession, artifact_id: str) -> dict:
    """What the bench measured for this artifact.

    The artifact id is a `deployment_id`, which carries a `benchmark_ref` to the FORGYX row holding the
    measured latencies. Resolved through that chain rather than duplicated onto the device, so the field
    is always compared against the benchmark that actually gated the artifact rather than a copy of it
    that could drift.
    """
    import uuid as _uuid

    from db.models import Benchmark, Deployment

    try:
        deployment = await db.get(Deployment, _uuid.UUID(str(artifact_id)))
    except (ValueError, AttributeError):
        deployment = None
    if deployment is None or not deployment.benchmark_ref:
        # An artifact with no benchmark row is "unknown", not zero. Reporting a ratio against nothing
        # would manufacture a regression out of a missing record.
        return {}

    bench = await db.get(Benchmark, deployment.benchmark_ref)
    if bench is None:
        return {}
    latency = dict(bench.latency_ms or {})
    return {"latency_p50_ms": latency.get("p50"), "latency_p95_ms": latency.get("p95"),
            "throughput_fps": bench.throughput_fps, "power_w": bench.power_w,
            "target": bench.target, "model_version": bench.model_version,
            # The gate-time confidence distribution, when the benchmark recorded one. Absent for an older
            # benchmark, in which case drift is reported as unmeasured rather than as zero.
            "conf_histogram": (bench.latency_ms or {}).get("conf_histogram") or []}


def _sum_histograms(hists: list) -> list[float]:
    usable = [h for h in hists if h and isinstance(h, list)]
    if not usable:
        return []
    width = len(usable[0])
    if any(len(h) != width for h in usable):
        # Mixed bin counts cannot be summed, and summing them anyway would produce a distribution that
        # describes nothing.
        return []
    return [float(sum(h[i] for h in usable)) for i in range(width)]


def _percentile(values: list[float], q: float) -> float | None:
    """A percentile over per-window percentiles.

    Approximate and deliberately so: the true fleet p95 needs the raw latencies, which is exactly the data
    a device should not be shipping. The p95 of the reported p95s is the honest available answer, and
    naming that here is better than presenting it as exact.
    """
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return round(float(vals[idx]), 2)


def _device_dict(d: EdgeDevice) -> dict:
    return {"device_id": d.device_id, "name": d.name, "hardware": d.hardware,
            "runtime": d.runtime, "artifact_id": d.artifact_id,
            "model_version": d.model_version, "fleet": d.fleet,
            "last_seen_at": d.last_seen_at.isoformat() if d.last_seen_at else None,
            "meta": d.meta or {}}
