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
from db.models import ConsentRecord, Frame, Object, PiiAudit
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
    return {"frames": len(frame_ids), "objects": n_objects, "pii_audits": n_pii,
            "frame_ids": [str(f) for f in frame_ids]}


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
        "blobs_deleted": blobs_deleted,
    }
    body["digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


async def erase_session(db: AsyncSession, session_id: uuid.UUID, *, reason: str,
                        dry_run: bool = False) -> dict:
    """Erase one session: its frames, their annotations, its PII audits, and the image blobs.

    dry_run reports the footprint without deleting, because the first thing anyone sensible does before an
    irreversible bulk delete is ask what it would touch.
    """
    sess = await db.get(DbSession, session_id)
    if sess is None:
        return {"error": "session not found", "session_id": str(session_id)}

    footprint = await _session_footprint(db, session_id)
    if dry_run:
        return {"dry_run": True, "session_id": str(session_id), **{k: v for k, v in footprint.items()
                                                                   if k != "frame_ids"}}

    # Blobs first: a database row is the only handle we have on its object-storage key, so deleting the row
    # before the blob orphans the file permanently, which is the opposite of erasure.
    store = get_object_store()
    blobs = 0
    uris = [u for (u,) in (await db.execute(
        select(Frame.img_uri).where(Frame.session_id == session_id))).all() if u]
    for uri in uris:
        try:
            store.remove(uri)
            blobs += 1
        except Exception as exc:  # noqa: BLE001 - a missing blob must not stop the erasure of the rest
            log.warning("retention.blob_delete_failed", uri=uri, error=str(exc))

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
