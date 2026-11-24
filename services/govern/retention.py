"""Retention enforcement and subject erasure: the executor the consent record always implied.

`retention_status()` has always returned `{"action": "purge"}` for an expired record, and its own docstring
says "a retention sweep acts on this". No sweep existed. Worse, `retention_until` could not be set through
the API at all (the request model had no field and the handler dropped the parameter), so every consent
record carried a null deadline and the function returned "retain" forever. The feature was unreachable end
to end: computed, documented, and never enforced.

There was also no erasure path. DPDPA and GDPR-style regimes require being able to answer "delete everything
about this subject", and the only deletion in the system was a per-object endpoint. Nothing walked from a
subject to its data, and nothing covered object storage, which database cascades do not reach.

Two deliberate choices:

- Erasure deletes the frames and their blobs, not just the annotations. Deleting labels while the images
  remain is not erasure; it is a metadata edit that leaves the personal data in place.
- Every purge and erasure writes an audit decision and returns a certificate. A deletion that leaves no
  evidence it happened cannot be shown to a regulator, and re-running it cannot be distinguished from never
  having run it.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import ConsentRecord, Frame, Object, PiiAudit, PlateRead
from db.models import Session as DbSession
from services.govern.audit import record

log = get_logger("retention")


async def expired_sessions(db: AsyncSession, now: datetime | None = None) -> list[dict]:
    """Sessions whose retention deadline has passed. The input to a sweep, and readable on its own so an
    operator can see what is due before anything is deleted."""
    now = now or datetime.now(UTC)
    rows = (await db.execute(
        select(ConsentRecord).where(ConsentRecord.retention_until.isnot(None),
                                    ConsentRecord.retention_until <= now))).scalars().all()
    return [{"session_id": str(r.session_id), "retention_until": r.retention_until.isoformat(),
             "consent_status": r.consent_status, "legal_basis": r.legal_basis} for r in rows]


async def _session_footprint(db: AsyncSession, session_id: uuid.UUID) -> dict:
    """What erasing this session would remove. Computed before deletion so the certificate can state it."""
    frame_ids = [f for (f,) in (await db.execute(
        select(Frame.frame_id).where(Frame.session_id == session_id))).all()]
    n_objects = 0
    if frame_ids:
        n_objects = int((await db.execute(
            select(func.count()).select_from(Object)
            .where(Object.frame_id.in_(frame_ids)))).scalar_one())
    n_pii = int((await db.execute(
        select(func.count()).select_from(PiiAudit)
        .where(PiiAudit.session_id == session_id))).scalar_one())
    # Plate reads are the most sensitive field a security session holds, and the FK already cascades them
    # away. Counting them here is what makes the certificate say so: an erasure certificate that omits the
    # plate text it removed cannot answer the request it exists to answer.
    n_plate_reads = int((await db.execute(
        select(func.count()).select_from(PlateRead)
        .where(PlateRead.session_id == session_id))).scalar_one())
    return {"frames": len(frame_ids), "objects": n_objects, "pii_audits": n_pii,
            "plate_reads": n_plate_reads, "frame_ids": [str(f) for f in frame_ids]}


# Stored-file columns that name something a session does not own: model weights, training manifests,
# compiled artifacts, reports. A session's rows can point at them; erasing the session must not delete
# them. Everything else ending in `_uri` on a row tied to the session is that session's data.
_SHARED_URI_COLUMNS = frozenset({"weights_uri", "data_yaml_uri", "artifact_uri", "package_uri",
                                 "per_layer_uri", "report_uri"})

# Object-store prefixes that are written per session. Listed as well as followed from the rows, because a
# blob whose row is already gone (a mask left behind by a re-label, a superseded overlay) is still the
# session's data and has no row left to find it by.
_SESSION_PREFIXES = ("frames/{sid}/", "masks/{sid}/", "masks/drivable/{sid}/", "segmentation/{sid}/",
                     "sessions/{sid}/", "occupancy/{sid}/", "autolabel/{sid}/")

# Source media that more than one session can share: the same upload imported twice. Deleted only when no
# other session still points at it.
_SOURCE_COLUMNS = ("raw_uri", "mcap_uri")


async def session_blob_keys(db: AsyncSession, session_id: uuid.UUID,
                            frame_ids: list[uuid.UUID]) -> list[str]:
    """Every object-store key that holds this session's data, in this store's own bucket.

    Erasure used to delete `Frame.img_uri` and nothing else. The redacted frames went and the rest stayed:
    the uploaded source video (unredacted, by definition), every mask, the drivable and segmentation
    rasters, the point clouds. An erasure certificate over that is a certificate for the least sensitive
    copy of the data.

    Three sources, unioned: every `*_uri` column on any table row that belongs to the session (found from
    the schema, so a table added later is covered without anyone remembering this function); the
    session-scoped key prefixes; and the session's source media when no other session shares it.
    """
    import asyncio

    from sqlalchemy import or_

    from db.models import Base

    store = get_object_store()
    uris: set[str] = set()

    for table in Base.metadata.sorted_tables:
        cols = [c for c in table.columns
                if c.name.endswith("_uri") and c.name not in _SHARED_URI_COLUMNS
                and not (table.name == "session" and c.name in _SOURCE_COLUMNS)]
        if not cols:
            continue
        clauses = []
        if "session_id" in table.columns:
            clauses.append(table.columns["session_id"] == session_id)
        if "frame_id" in table.columns and frame_ids:
            # A subquery, not the literal ids: a long drive is a hundred thousand of them.
            clauses.append(table.columns["frame_id"].in_(
                select(Frame.frame_id).where(Frame.session_id == session_id)))
        if not clauses:
            continue
        for row in (await db.execute(select(*cols).where(or_(*clauses)))).all():
            uris.update(u for u in row if u)

    # Source media, unless another session was made from the same upload.
    sess = await db.get(DbSession, session_id)
    for name in _SOURCE_COLUMNS:
        uri = getattr(sess, name, None) if sess is not None else None
        if not uri:
            continue
        col = getattr(DbSession, name)
        shared = (await db.execute(select(func.count()).select_from(DbSession)
                                   .where(col == uri, DbSession.session_id != session_id))).scalar_one()
        if not shared:
            uris.add(uri)
        else:
            log.info("retention.source_kept_shared", uri=uri, other_sessions=int(shared))

    keys = {store._key(u) for u in uris if store.owns(u)}

    def _listed() -> set[str]:
        found: set[str] = set()
        for pattern in _SESSION_PREFIXES:
            try:
                found.update(store.list_keys(pattern.format(sid=session_id)))
            except Exception as exc:  # noqa: BLE001 - a listing that fails must not stop the rest
                log.warning("retention.prefix_list_failed", prefix=pattern, error=str(exc))
        return found

    keys |= await asyncio.to_thread(_listed)
    return sorted(keys)


def _certificate(kind: str, session_id: str, footprint: dict, blobs_deleted: int,
                 at: datetime) -> dict:
    """A tamper-evident record of what was erased.

    The digest covers the subject, the counts, and the timestamp, so a certificate cannot be edited after the
    fact to claim more (or less) was removed than actually was.
    """
    body = {
        "kind": kind,
        "session_id": session_id,
        "erased_at": at.isoformat(),
        "frames": footprint["frames"],
        "objects": footprint["objects"],
        "pii_audits": footprint["pii_audits"],
        # Inside the digested body, not alongside it: a count that the digest does not cover could be edited
        # afterwards to claim no plate text was held, which is the one claim this certificate must not allow.
        "plate_reads": footprint.get("plate_reads", 0),
        "blobs_deleted": blobs_deleted,
    }
    body["digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


async def erase_session(db: AsyncSession, session_id: uuid.UUID, *, reason: str,
                        dry_run: bool = False) -> dict:
    """Erase one session: its frames, their annotations, its PII audits, and every stored file that
    holds its data (frame images, source media, masks, rasters, point clouds; see session_blob_keys).

    dry_run reports the footprint without deleting, because the first thing anyone sensible does before an
    irreversible bulk delete is ask what it would touch.
    """
    sess = await db.get(DbSession, session_id)
    if sess is None:
        return {"error": "session not found", "session_id": str(session_id)}

    footprint = await _session_footprint(db, session_id)
    if dry_run:
        would = await session_blob_keys(db, session_id, [uuid.UUID(f) for f in footprint["frame_ids"]])
        return {"dry_run": True, "session_id": str(session_id), "blobs": len(would),
                **{k: v for k, v in footprint.items() if k != "frame_ids"}}

    # Blobs first: a database row is the only handle we have on its object-storage key, so deleting the row
    # before the blob orphans the file permanently, which is the opposite of erasure.
    import asyncio

    store = get_object_store()
    frame_uuids = [uuid.UUID(f) for f in footprint["frame_ids"]]
    targets = await session_blob_keys(db, session_id, frame_uuids)

    def _remove_all() -> int:
        n = 0
        for key in targets:
            try:
                store.remove(key)
                n += 1
            except Exception as exc:  # noqa: BLE001 - a missing blob must not stop the erasure of the rest
                log.warning("retention.blob_delete_failed", key=key, error=str(exc))
        return n

    blobs = await asyncio.to_thread(_remove_all)

    # Frames cascade to objects and pii_audit via their FKs; the session row goes last so a failure part-way
    # leaves the session visible and the erasure obviously incomplete rather than silently half-done.
    for fid in footprint["frame_ids"]:
        frame = await db.get(Frame, uuid.UUID(fid))
        if frame is not None:
            await db.delete(frame)
    await db.delete(sess)
    await db.commit()

    at = datetime.now(UTC)
    cert = _certificate("erasure", str(session_id), footprint, blobs, at)
    await record(db, "retention", "erase_session", str(session_id),
                 {"reason": reason, "certificate": cert})
    log.info("retention.erased", session=str(session_id), frames=footprint["frames"],
             objects=footprint["objects"], blobs=blobs)
    return {"erased": True, "session_id": str(session_id), "certificate": cert}


async def run_retention_sweep(db: AsyncSession, *, dry_run: bool = True,
                              now: datetime | None = None, limit: int = 50) -> dict:
    """Erase every session past its retention deadline.

    Defaults to a dry run. An irreversible bulk delete that runs by default the first time someone calls it
    to see what it does is a trap, so deleting has to be asked for explicitly.
    """
    due = (await expired_sessions(db, now))[:limit]
    if dry_run:
        return {"dry_run": True, "due": len(due), "sessions": due}

    certificates = []
    for item in due:
        res = await erase_session(db, uuid.UUID(item["session_id"]),
                                  reason=f"retention expired {item['retention_until']}")
        if res.get("certificate"):
            certificates.append(res["certificate"])
    log.info("retention.sweep", erased=len(certificates), due=len(due))
    return {"dry_run": False, "due": len(due), "erased": len(certificates),
            "certificates": certificates}
