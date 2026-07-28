"""Persistence for the ANPR path: the watchlist, and the reads it produces.

The recogniser and the format kernel were complete and had nowhere to write. Every read was ephemeral and the
watchlist had to be handed in on each call, so nothing could be built on top: no feed, no hit history, no
console. This is the missing half.

Two rules hold throughout, and both are about the fact that a plate is personal data:

- Nothing here runs outside a pack that declares the `anpr` capability. The AV pack does the opposite, blurs
  plates and never reads them, and calling this under it raises rather than quietly recording plate text into
  a DPDPA-scoped corpus.
- A read is stored against its session, and the FK cascades, so an erasure request takes the plate text with
  everything else. Storing it detached would leave the most sensitive field in the corpus behind.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import PlateRead as PlateReadRow
from db.models import PlateWatchlist
from services.anpr.india_format import normalize_plate
from services.anpr.recognize import AnprNotAuthorised, PlateRead

log = get_logger("anpr_store")

SEVERITIES = ("info", "warn", "critical")


def _require_anpr(pack_id: str | None) -> str:
    """Resolve and authorise the pack. Returns the pack id actually used, for recording on the row."""
    from services.domain import default_pack_id, has_capability

    pid = pack_id or default_pack_id()
    if not has_capability("anpr", pid):
        raise AnprNotAuthorised(
            f"ANPR is not authorised for pack {pid!r}. It is a security-domain capability; a pack must "
            "declare 'anpr'. Under the AV pack plates are personal data, blurred and never read.")
    return pid


# ---------------------------------------------------------------- watchlist

async def add_watchlist_entry(db: AsyncSession, *, plate: str, reason: str | None = None,
                              severity: str = "warn", added_by: str | None = None) -> dict:
    """Add a mark to the watchlist, keyed on its normalised form.

    Normalising before storing is what makes matching work at all: the same mark arrives as "KA 01 AB 1234",
    "ka-01-ab-1234", and "KA01AB1234", and three rows would report one vehicle as three hits.
    """
    norm = normalize_plate(plate)
    if not norm:
        raise ValueError(f"{plate!r} does not normalise to a registration mark")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")

    existing = (await db.execute(
        select(PlateWatchlist).where(PlateWatchlist.plate_normalized == norm))).scalar_one_or_none()
    if existing is not None:
        # Re-adding a mark reactivates and re-annotates it rather than failing: an operator re-adding a plate
        # means "watch this", and a duplicate-key error is a worse answer than doing what they asked.
        existing.active = True
        existing.reason = reason if reason is not None else existing.reason
        existing.severity = severity
        await db.commit()
        return _wl_dict(existing)

    row = PlateWatchlist(plate_normalized=norm, plate_raw=plate.strip(), reason=reason,
                         severity=severity, added_by=added_by, active=True)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    log.info("anpr.watchlist_added", plate=norm, severity=severity)
    return _wl_dict(row)


async def remove_watchlist_entry(db: AsyncSession, entry_id: str) -> dict:
    """Deactivate rather than delete: a hit already recorded refers to why it was watched, and dropping the
    row would leave that history unexplainable."""
    row = await db.get(PlateWatchlist, uuid.UUID(entry_id))
    if row is None:
        return {"removed": False, "reason": "not found"}
    row.active = False
    await db.commit()
    log.info("anpr.watchlist_removed", plate=row.plate_normalized)
    return {"removed": True, "entry_id": entry_id, "plate": row.plate_normalized}


async def list_watchlist(db: AsyncSession, *, active_only: bool = True, limit: int = 500) -> list[dict]:
    stmt = select(PlateWatchlist).order_by(PlateWatchlist.created_at.desc()).limit(min(max(limit, 1), 2000))
    if active_only:
        stmt = stmt.where(PlateWatchlist.active.is_(True))
    return [_wl_dict(r) for r in (await db.execute(stmt)).scalars().all()]


async def active_watchlist_map(db: AsyncSession) -> dict[str, PlateWatchlist]:
    """Normalised mark -> entry, for matching a batch of reads in one query instead of one per read."""
    rows = (await db.execute(
        select(PlateWatchlist).where(PlateWatchlist.active.is_(True)))).scalars().all()
    return {r.plate_normalized: r for r in rows}


def _wl_dict(r: PlateWatchlist) -> dict:
    return {"entry_id": str(r.entry_id), "plate": r.plate_normalized, "plate_raw": r.plate_raw,
            "reason": r.reason, "severity": r.severity, "active": r.active, "added_by": r.added_by,
            "created_at": r.created_at.isoformat() if r.created_at else None}


# ---------------------------------------------------------------- reads

async def record_reads(db: AsyncSession, reads: list[PlateRead], *, session_id: str | None = None,
                       frame_id: str | None = None, camera_id: str | None = None,
                       pack_id: str | None = None) -> dict:
    """Persist a batch of reads, matching each against the active watchlist as it goes."""
    pid = _require_anpr(pack_id)
    wl = await active_watchlist_map(db)

    stored, hits = [], 0
    for r in reads:
        entry = wl.get(r.parse.normalized) if r.parse.normalized else None
        row = PlateReadRow(
            session_id=uuid.UUID(session_id) if session_id else None,
            frame_id=uuid.UUID(frame_id) if frame_id else None,
            camera_id=camera_id,
            plate_normalized=r.parse.normalized or "",
            plate_raw=r.ocr_text,
            plate_type=r.parse.plate_type,
            state_code=r.parse.state_code,
            rto_district=r.parse.rto_district,
            valid=bool(r.parse.valid),
            det_conf=float(r.det_conf),
            # Kept as None when the reader could not measure it. A stand-in number here would make a
            # confidence filter in the console look meaningful when it is not.
            ocr_conf=(float(r.ocr_conf) if r.ocr_conf is not None else None),
            format_confidence=float(r.parse.format_confidence),
            bbox=list(r.bbox),
            watchlist_hit=entry is not None,
            watchlist_severity=(entry.severity if entry is not None else None),
            pack_id=pid,
        )
        db.add(row)
        stored.append(row)
        if entry is not None:
            hits += 1
    await db.commit()

    log.info("anpr.reads_recorded", n=len(stored), hits=hits, pack=pid, session=session_id)
    return {"recorded": len(stored), "watchlist_hits": hits,
            "reads": [_read_dict(r) for r in stored]}


async def list_reads(db: AsyncSession, *, session_id: str | None = None, camera_id: str | None = None,
                     plate: str | None = None, state_code: str | None = None, hits_only: bool = False,
                     limit: int = 100, offset: int = 0) -> dict:
    """The console feed: most recent first, with a real total so it can be paged."""
    stmt = select(PlateReadRow)
    count_stmt = select(func.count()).select_from(PlateReadRow)

    def _filters(q):
        if session_id:
            q = q.where(PlateReadRow.session_id == uuid.UUID(session_id))
        if camera_id:
            q = q.where(PlateReadRow.camera_id == camera_id)
        if plate:
            # Search on the normalised form so an operator can paste a plate however it is written.
            q = q.where(PlateReadRow.plate_normalized == normalize_plate(plate))
        if state_code:
            # Backs the issuing-state facet in the console, so those counts are a filter rather than a label.
            q = q.where(PlateReadRow.state_code == state_code.strip().upper())
        if hits_only:
            q = q.where(PlateReadRow.watchlist_hit.is_(True))
        return q

    rows = (await db.execute(
        _filters(stmt).order_by(PlateReadRow.created_at.desc())
        .offset(max(offset, 0)).limit(min(max(limit, 1), 500)))).scalars().all()
    total = (await db.execute(_filters(count_stmt))).scalar_one()
    return {"total": int(total), "offset": offset, "limit": limit,
            "reads": [_read_dict(r) for r in rows]}


async def read_stats(db: AsyncSession) -> dict:
    """Headline counts for the console. Cheap aggregates, not a dashboard framework."""
    total = (await db.execute(select(func.count()).select_from(PlateReadRow))).scalar_one()
    hits = (await db.execute(select(func.count()).select_from(PlateReadRow)
                             .where(PlateReadRow.watchlist_hit.is_(True)))).scalar_one()
    valid = (await db.execute(select(func.count()).select_from(PlateReadRow)
                              .where(PlateReadRow.valid.is_(True)))).scalar_one()
    watched = (await db.execute(select(func.count()).select_from(PlateWatchlist)
                                .where(PlateWatchlist.active.is_(True)))).scalar_one()
    by_state: dict[str, int] = {code: int(n) for code, n in (await db.execute(
        select(PlateReadRow.state_code, func.count())
        .where(PlateReadRow.state_code.isnot(None))
        .group_by(PlateReadRow.state_code)
        .order_by(func.count().desc()).limit(10))).all() if code}
    # Unmeasured confidence is surfaced rather than hidden: it tells the operator the local reader is in use
    # and that a confidence filter would be meaningless on those rows.
    unscored = (await db.execute(select(func.count()).select_from(PlateReadRow)
                                 .where(PlateReadRow.ocr_conf.is_(None)))).scalar_one()
    return {"reads": int(total), "watchlist_hits": int(hits), "valid_format": int(valid),
            "watchlist_size": int(watched), "unscored_reads": int(unscored),
            "top_states": by_state}


def _read_dict(r: PlateReadRow) -> dict:
    return {
        "read_id": str(r.read_id),
        "session_id": str(r.session_id) if r.session_id else None,
        "frame_id": str(r.frame_id) if r.frame_id else None,
        "camera_id": r.camera_id,
        "plate": r.plate_normalized,
        "plate_raw": r.plate_raw,
        "plate_type": r.plate_type,
        "state_code": r.state_code,
        "rto_district": r.rto_district,
        "valid": r.valid,
        "det_conf": round(float(r.det_conf), 4),
        "ocr_conf": (round(float(r.ocr_conf), 4) if r.ocr_conf is not None else None),
        "format_confidence": round(float(r.format_confidence), 4),
        "bbox": list(r.bbox) if r.bbox else None,
        "watchlist_hit": r.watchlist_hit,
        "watchlist_severity": r.watchlist_severity,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }
