"""Incidents: one thing that happened, assembled from everything that evidenced it.

A plate read, a zone crossing and a person track were three unrelated rows about the same van arriving at
the same gate at the same moment, and an operator had to build the event in their head from three screens.
Nothing in the system said they were one thing, so nothing could be acknowledged, escalated, or counted.

Stitching is by camera and time window, not by identity. Identity is what you get out of an incident, not
what you need to form one: insisting on a plate or a signature before grouping would drop exactly the
events where the plate was unreadable, which are the ones worth looking at.

Two windows, and the difference matters. `STITCH_WINDOW_NS` is how close two pieces of evidence must be to
be the same event; `DEDUPE_WINDOW_NS` is how long before the same rule on the same subject is a new event
rather than the same one continuing. The first is short because a gate crossing and its plate read are
simultaneous; the second is long because a van parked across a line should not raise an incident a minute.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import CameraZone, SecurityIncident

log = get_logger("sec_incidents")

STITCH_WINDOW_NS = 5 * 1_000_000_000        # 5s: a crossing and its plate read are simultaneous
DEDUPE_WINDOW_NS = 120 * 1_000_000_000      # 2min: a parked van is one incident, not sixty

KINDS = ("zone_crossing", "watchlist_hit", "dwell", "reid_match", "abandoned_object", "manual")


class IncidentError(Exception):
    """An incident operation refused."""


def _require_sec(pack_id: str | None) -> str:
    """Scene analytics is a security capability, same gate as ANPR.

    An AV deployment must not be able to draw a tripwire on a dashcam and start recording who crossed it.
    """
    from services.anpr.recognize import AnprNotAuthorised
    from services.domain import default_pack_id, has_capability

    pid = pack_id or default_pack_id()
    if not has_capability("static_camera", pid):
        raise AnprNotAuthorised(
            f"scene analytics is not authorised for pack {pid!r}. Zones, incidents and re-identification "
            "are security-domain capabilities; a pack must declare 'static_camera'.")
    return pid


# ---------------------------------------------------------------- zones

async def create_zone(db: AsyncSession, *, camera_id: str, name: str, points: list,
                      kind: str = "area", rule: str = "enter", classes: list[str] | None = None,
                      dwell_seconds: float | None = None, severity: str = "warn",
                      session_id: str | None = None, created_by: str | None = None,
                      pack_id: str | None = None) -> dict:
    from packs.sec.zones import validate_zone

    pid = _require_sec(pack_id)
    try:
        validate_zone(kind, rule, points, dwell_seconds)
    except ValueError as exc:
        raise IncidentError(str(exc)) from exc

    row = CameraZone(camera_id=camera_id, session_id=uuid.UUID(session_id) if session_id else None,
                     name=name, kind=kind, points=list(points), rule=rule,
                     dwell_seconds=dwell_seconds, classes=list(classes or []),
                     severity=severity, created_by=created_by, pack_id=pid, active=True)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    log.info("sec.zone_created", camera=camera_id, name=name, kind=kind, rule=rule)
    return _zone_dict(row)


async def list_zones(db: AsyncSession, *, camera_id: str | None = None,
                     active_only: bool = True) -> dict:
    stmt = select(CameraZone).order_by(CameraZone.created_at.desc())
    if camera_id:
        stmt = stmt.where(CameraZone.camera_id == camera_id)
    if active_only:
        stmt = stmt.where(CameraZone.active.is_(True))
    return {"zones": [_zone_dict(z) for z in (await db.execute(stmt)).scalars().all()]}


async def delete_zone(db: AsyncSession, zone_id: str) -> dict:
    """Deactivate rather than delete: incidents point at the zone that raised them, and dropping the row
    would leave a history of alerts nobody can explain."""
    z = await db.get(CameraZone, uuid.UUID(zone_id))
    if z is None:
        return {"removed": False, "reason": "not found"}
    z.active = False
    await db.commit()
    return {"removed": True, "zone_id": zone_id, "name": z.name}


# ---------------------------------------------------------------- evaluation

async def evaluate_session(db: AsyncSession, session_id: str, *,
                           pack_id: str | None = None) -> dict:
    """Run every active zone for this session's camera over its tracks, and raise what fires."""
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from packs.sec.zones import evaluate_track
    from services.autolabel.ontology import get_ontology

    sess = await db.get(DbSession, uuid.UUID(session_id))
    if sess is None:
        raise IncidentError(f"session {session_id} not found")
    pid = _require_sec(pack_id or sess.pack_id)

    cam = (await db.execute(
        select(Frame.cam_id).where(Frame.session_id == sess.session_id).limit(1))).scalar_one_or_none()
    if not cam:
        return {"session_id": session_id, "zones": 0, "incidents": 0,
                "detail": "the session has no frames"}

    zones = (await db.execute(
        select(CameraZone).where(CameraZone.camera_id == cam,
                                 CameraZone.active.is_(True)))).scalars().all()
    if not zones:
        return {"session_id": session_id, "camera_id": cam, "zones": 0, "incidents": 0,
                "detail": "no active zones on this camera"}

    onto = get_ontology()
    rows = (await db.execute(
        select(Object, Frame.ts_ns)
        .join(Frame, Object.frame_id == Frame.frame_id)
        .where(Frame.session_id == sess.session_id, Object.state != "rejected")
        .order_by(Frame.ts_ns))).all()

    by_track: dict[str, list[dict]] = {}
    for obj, ts in rows:
        # An untracked detection is keyed on its own id, so it is a track of one. Dropping it would make
        # zone rules silently blind wherever tracking failed, which is where they matter most.
        key = str(obj.track_id) if obj.track_id else f"obj:{obj.object_id}"
        by_track.setdefault(key, []).append({
            "track_id": str(obj.track_id) if obj.track_id else None,
            "object_id": str(obj.object_id), "frame_id": str(obj.frame_id),
            "class_name": onto.by_id(obj.class_id).name, "bbox": list(obj.bbox), "ts_ns": int(ts)})

    raised = 0
    for zone in zones:
        zd = _zone_dict(zone)
        for samples in by_track.values():
            for crossing in evaluate_track(zd, samples):
                created = await raise_incident(
                    db, kind="dwell" if crossing.rule == "dwell" else "zone_crossing",
                    camera_id=cam, session_id=session_id, zone_id=str(zone.zone_id),
                    severity=crossing.severity,
                    title=f"{crossing.class_name} {crossing.rule} {zone.name}",
                    summary=None, ts_ns=crossing.ts_ns,
                    evidence={"track_id": crossing.track_id, "rule": crossing.rule,
                              **crossing.detail},
                    pack_id=pid)
                if created.get("created"):
                    raised += 1

    log.info("sec.session_evaluated", session=session_id, camera=cam, zones=len(zones), raised=raised)
    return {"session_id": session_id, "camera_id": cam, "zones": len(zones),
            "tracks": len(by_track), "incidents": raised}


# ---------------------------------------------------------------- incidents

async def raise_incident(db: AsyncSession, *, kind: str, title: str, ts_ns: int,
                         camera_id: str | None = None, session_id: str | None = None,
                         zone_id: str | None = None, severity: str = "warn",
                         summary: str | None = None, evidence: dict | None = None,
                         plate: str | None = None, person_identity: str | None = None,
                         pack_id: str | None = None, dedupe: bool = True) -> dict:
    """Raise one, or fold it into an open incident it belongs to.

    Folding rather than creating is what keeps the board readable: a van sitting across a tripwire produces
    one incident that grows, not one per frame.
    """
    pid = _require_sec(pack_id)
    if kind not in KINDS:
        raise IncidentError(f"kind must be one of {KINDS}")

    if dedupe:
        existing = await _open_match(db, kind, camera_id, zone_id, plate, person_identity, ts_ns)
        if existing is not None:
            existing.end_ts_ns = max(int(existing.end_ts_ns), int(ts_ns))
            merged = dict(existing.evidence or {})
            for k, v in (evidence or {}).items():
                if k in merged and merged[k] != v:
                    # Values accumulate into a list rather than overwriting: an incident spanning three
                    # tracks should name all three, not just the last one seen.
                    prev = merged[k] if isinstance(merged[k], list) else [merged[k]]
                    merged[k] = [*prev, v] if v not in prev else prev
                else:
                    merged[k] = v
            merged["events"] = int(merged.get("events") or 1) + 1
            existing.evidence = merged
            await db.commit()
            return {"created": False, "merged_into": str(existing.incident_id),
                    "incident": _dict(existing)}

    row = SecurityIncident(
        camera_id=camera_id, session_id=uuid.UUID(session_id) if session_id else None,
        zone_id=uuid.UUID(zone_id) if zone_id else None, kind=kind, severity=severity,
        title=title, summary=summary, start_ts_ns=int(ts_ns), end_ts_ns=int(ts_ns),
        evidence={**(evidence or {}), "events": 1}, plate=plate,
        person_identity=person_identity, pack_id=pid, status="open")
    db.add(row)
    await db.commit()
    await db.refresh(row)

    from services.notify import notify

    await notify(db, kind="watchlist_hit" if kind == "watchlist_hit" else "issue_opened",
                 severity=severity, title=title, body=summary,
                 href=f"/labeloxsec/incidents?id={row.incident_id}",
                 subject_type="incident", subject_id=str(row.incident_id), supersede=False)
    log.info("sec.incident_raised", kind=kind, camera=camera_id, severity=severity)
    return {"created": True, "incident": _dict(row)}


async def _open_match(db: AsyncSession, kind: str, camera_id: str | None, zone_id: str | None,
                      plate: str | None, person_identity: str | None,
                      ts_ns: int) -> SecurityIncident | None:
    stmt = (select(SecurityIncident)
            .where(SecurityIncident.kind == kind, SecurityIncident.status == "open",
                   SecurityIncident.end_ts_ns >= int(ts_ns) - DEDUPE_WINDOW_NS,
                   SecurityIncident.start_ts_ns <= int(ts_ns) + DEDUPE_WINDOW_NS)
            .order_by(SecurityIncident.end_ts_ns.desc()).limit(1))
    if camera_id:
        stmt = stmt.where(SecurityIncident.camera_id == camera_id)
    if zone_id:
        stmt = stmt.where(SecurityIncident.zone_id == uuid.UUID(zone_id))
    if plate:
        stmt = stmt.where(SecurityIncident.plate == plate)
    if person_identity:
        stmt = stmt.where(SecurityIncident.person_identity == person_identity)
    return (await db.execute(stmt)).scalars().first()


async def stitch_plate_reads(db: AsyncSession, session_id: str, *,
                             pack_id: str | None = None) -> dict:
    """Attach plate reads to the incidents that happened at the same moment on the same camera.

    This is the join an operator was doing in their head: a crossing at 14:02:11 and a plate read at
    14:02:11 on the same camera are one van, and the incident should say which van.
    """
    from db.models import Frame
    from db.models import PlateRead as PlateReadRow

    _require_sec(pack_id)
    reads = (await db.execute(
        select(PlateReadRow, Frame.ts_ns)
        .join(Frame, PlateReadRow.frame_id == Frame.frame_id, isouter=True)
        .where(PlateReadRow.session_id == uuid.UUID(session_id)))).all()

    attached = 0
    for read, ts in reads:
        if ts is None:
            continue
        inc = (await db.execute(
            select(SecurityIncident)
            .where(SecurityIncident.camera_id == read.camera_id,
                   SecurityIncident.start_ts_ns <= int(ts) + STITCH_WINDOW_NS,
                   SecurityIncident.end_ts_ns >= int(ts) - STITCH_WINDOW_NS,
                   SecurityIncident.plate.is_(None))
            .order_by(SecurityIncident.start_ts_ns.desc()).limit(1))).scalars().first()
        if inc is None:
            continue
        inc.plate = read.plate_normalized
        ev = dict(inc.evidence or {})
        ev["plate_read_id"] = str(read.read_id)
        ev["plate_valid"] = bool(read.valid)
        inc.evidence = ev
        if read.watchlist_hit:
            # A watchlist hit changes what the incident is, not just what it knows: the same crossing with
            # a watched plate is a different event and must escalate.
            inc.severity = read.watchlist_severity or "critical"
            inc.title = f"{inc.title} (watched plate {read.plate_normalized})"
        attached += 1
    await db.commit()
    log.info("sec.plates_stitched", session=session_id, attached=attached)
    return {"session_id": session_id, "reads": len(reads), "attached": attached}


async def list_incidents(db: AsyncSession, *, camera_id: str | None = None, kind: str | None = None,
                         status: str | None = None, severity: str | None = None,
                         since_hours: int | None = None, limit: int = 100,
                         offset: int = 0) -> dict:
    stmt = select(SecurityIncident)
    count_stmt = select(func.count()).select_from(SecurityIncident)

    def _f(q):
        if camera_id:
            q = q.where(SecurityIncident.camera_id == camera_id)
        if kind:
            q = q.where(SecurityIncident.kind == kind)
        if status:
            q = q.where(SecurityIncident.status == status)
        if severity:
            q = q.where(SecurityIncident.severity == severity)
        if since_hours:
            q = q.where(SecurityIncident.created_at >= datetime.now(UTC) - timedelta(hours=since_hours))
        return q

    rows = (await db.execute(
        _f(stmt).order_by(SecurityIncident.start_ts_ns.desc())
        .offset(max(offset, 0)).limit(min(max(limit, 1), 500)))).scalars().all()
    total = (await db.execute(_f(count_stmt))).scalar_one()
    return {"total": int(total), "offset": offset, "limit": limit,
            "incidents": [_dict(r) for r in rows]}


async def acknowledge(db: AsyncSession, incident_id: str, *, by: str | None = None,
                      close: bool = False) -> dict:
    inc = await db.get(SecurityIncident, uuid.UUID(incident_id))
    if inc is None:
        raise IncidentError("incident not found")
    inc.status = "closed" if close else "ack"
    inc.acknowledged_by = by
    await db.commit()
    return _dict(inc)


def _zone_dict(z: CameraZone) -> dict:
    return {"zone_id": str(z.zone_id), "camera_id": z.camera_id,
            "session_id": str(z.session_id) if z.session_id else None,
            "name": z.name, "kind": z.kind, "points": z.points or [], "rule": z.rule,
            "dwell_seconds": z.dwell_seconds, "classes": z.classes or [],
            "severity": z.severity, "active": z.active, "created_by": z.created_by,
            "created_at": z.created_at.isoformat() if z.created_at else None}


def _dict(i: SecurityIncident) -> dict:
    return {"incident_id": str(i.incident_id), "camera_id": i.camera_id,
            "session_id": str(i.session_id) if i.session_id else None,
            "zone_id": str(i.zone_id) if i.zone_id else None,
            "kind": i.kind, "severity": i.severity, "title": i.title, "summary": i.summary,
            "start_ts_ns": int(i.start_ts_ns), "end_ts_ns": int(i.end_ts_ns),
            "duration_s": round(max(0, int(i.end_ts_ns) - int(i.start_ts_ns)) / 1e9, 2),
            "evidence": i.evidence or {}, "plate": i.plate,
            "person_identity": i.person_identity, "status": i.status,
            "acknowledged_by": i.acknowledged_by,
            "created_at": i.created_at.isoformat() if i.created_at else None}
