"""PII backfill for frames ingested before the anonymization gate existed.

The gate blurs faces and plates at ingest, so every new frame is clean before it reaches storage. But
frames captured earlier (or through a path that bypassed the gate) can still hold an un-blurred plate,
which is the DPDPA exposure visible in the editor. This re-runs the same anonymizer over frames that have
no PII audit, blurs in place, overwrites the stored image at its existing key, and writes the audit -- so
the corpus becomes uniformly clean without a full re-ingest. Idempotent: a frame that already has an audit
is skipped.
"""

from __future__ import annotations

import uuid

import cv2
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, PiiAudit
from db.session import get_sessionmaker
from services.anonymize.anonymizer import get_anonymizer
from services.recall.backends import load_image_bgr

log = get_logger("anonymize.backfill")


async def _audit_of(db: AsyncSession, frame_id: uuid.UUID) -> PiiAudit | None:
    return (await db.execute(
        select(PiiAudit).where(PiiAudit.frame_id == frame_id).limit(1))).scalars().first()


async def _has_audit(db: AsyncSession, frame_id: uuid.UUID) -> bool:
    return await _audit_of(db, frame_id) is not None


async def backfill_frame(db: AsyncSession, store, anonymizer, frame: Frame,
                         *, redo: bool = False) -> dict | None:
    """Blur a single frame in place and record its audit. Returns counts, or None if it was skipped.

    `redo` re-processes a frame that already has an audit, replacing it. That is not the same as running the
    gate twice for no reason: an audit records the method version that produced it, and a frame processed by
    a method that could not see small plates needs processing again. Blurring is idempotent in the way that
    matters, since a region already blurred simply blurs again.
    """
    existing = await _audit_of(db, frame.frame_id)
    if existing is not None and not redo:
        return None
    if existing is not None and existing.method_version == anonymizer.method_version:
        return None                      # already processed by the current method; nothing to redo
    try:
        img = load_image_bgr(store, frame.img_uri)
    except Exception as exc:  # noqa: BLE001
        log.warning("backfill.load_failed", frame_id=str(frame.frame_id), error=str(exc))
        return None
    pii = anonymizer.anonymize(img)          # blurs faces + plates in place
    if pii.n_faces or pii.n_plates:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return None
        _bucket, key = store.parse_uri(frame.img_uri)
        store.put_bytes(key, buf.tobytes(), "image/jpeg")   # overwrite the stored image at its own key
    if existing is not None:
        await db.delete(existing)        # one audit per frame; the new method's answer replaces the old
        await db.flush()
    db.add(PiiAudit(frame_id=frame.frame_id, session_id=frame.session_id, n_faces=pii.n_faces,
                    n_plates=pii.n_plates, regions=pii.regions, method_version=pii.method_version,
                    ts_ns=frame.ts_ns))
    return {"n_faces": pii.n_faces, "n_plates": pii.n_plates}


async def backfill_unaudited(limit: int = 500, session_id: str | None = None,
                             stale_method: bool = False) -> dict:
    """Backfill frames the gate has not cleaned (bounded). Faces/plates blurred, images overwritten in
    place, audits written. Reports how many frames were cleaned and how many plates/faces were caught.

    `stale_method` selects the other kind of unclean frame: one that WAS processed, by a method that has
    since been superseded. This corpus holds 41,718 of those. Every one was run at the detector's default
    inference size, which downsampled a 1920x1080 frame by three and made a plate roughly 150x40 pixels
    arrive as 50x13. Sampled afterwards at native resolution, the same detector finds a plate on 51.4% of
    the frames it had recorded as plate-free. An audit that says a frame is clean is only as good as the
    method that produced it, which is why the method version is recorded on the row and matched here.
    """
    store = get_object_store()
    anonymizer = get_anonymizer()             # raises loudly if a required detector is unavailable
    maker = get_sessionmaker()
    totals = {"frames": 0, "cleaned": 0, "n_faces": 0, "n_plates": 0}

    async with maker() as db:
        if stale_method:
            q = (select(Frame).join(PiiAudit, PiiAudit.frame_id == Frame.frame_id)
                 .where(PiiAudit.method_version != anonymizer.method_version))
        else:
            q = (select(Frame).outerjoin(PiiAudit, PiiAudit.frame_id == Frame.frame_id)
                 .where(PiiAudit.frame_id.is_(None)))
        if session_id:
            q = q.where(Frame.session_id == uuid.UUID(session_id))
        frames = (await db.execute(q.limit(limit))).scalars().all()

    for frame in frames:
        async with maker() as db:
            res = await backfill_frame(db, store, anonymizer, frame, redo=stale_method)
            if res is None:
                continue
            await db.commit()
        totals["frames"] += 1
        if res["n_faces"] or res["n_plates"]:
            totals["cleaned"] += 1
        totals["n_faces"] += res["n_faces"]
        totals["n_plates"] += res["n_plates"]

    log.info("backfill.done", **totals)
    return totals
