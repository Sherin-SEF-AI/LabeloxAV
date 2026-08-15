"""PII backfill for frames ingested before the anonymization gate existed.

The gate blurs faces and plates at ingest, so every new frame is clean before it reaches storage. But
frames captured earlier (or through a path that bypassed the gate) can still hold an un-blurred plate,
which is the DPDPA exposure visible in the editor. This re-runs the same anonymizer over frames that have
no PII audit, blurs in place, overwrites the stored image at its existing key, and writes the audit -- so
the corpus becomes uniformly clean without a full re-ingest. Idempotent: a frame that already has an audit
is skipped.
"""

from __future__ import annotations

import asyncio
import uuid

import cv2
from botocore.exceptions import ClientError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, PiiAudit
from db.session import get_sessionmaker
from services.anonymize.anonymizer import get_anonymizer
from services.recall.backends import load_image_bgr

log = get_logger("anonymize.backfill")

# Seconds to wait before each further attempt at a rejected write. These are long because botocore has
# already spent its own three fast retries by the time the call raises: a store that is still refusing after
# that is asking for a slower caller, not a quicker one.
_WRITE_RETRY_DELAYS_S = (2.0, 8.0, 30.0)

# Codes that mean "not now" rather than "not ever". A backfill overwriting tens of thousands of objects is
# exactly the shape of traffic a store sheds under pressure, and MinIO answered a sustained 8 writes/second
# with SlowDownWrite after 31,976 frames.
_WRITE_PRESSURE_CODES = frozenset({
    "SlowDown", "SlowDownWrite", "SlowDownRead", "ServiceUnavailable", "InternalError", "RequestTimeout",
})

# A store that refuses this many frames in a row is down or full, and the remaining frames will not fare
# better. Stopping says so; grinding through 10,000 more failures buries it.
_CONSECUTIVE_FAILURE_LIMIT = 20


class StoreRefused(RuntimeError):
    """The store would not accept a frame's redacted image, after waiting.

    Distinct from a skip, because a skipped frame is finished and a refused one still carries an unredacted
    plate. The run continues past it (one frame is not the corpus) but has to be able to count them.
    """


def _is_write_pressure(exc: BaseException) -> bool:
    if not isinstance(exc, ClientError):
        return False
    return exc.response.get("Error", {}).get("Code", "") in _WRITE_PRESSURE_CODES


async def _put_with_backoff(store, key: str, data: bytes, content_type: str) -> bool:
    """Write one object, waiting out a store that is shedding load. False if it never took it.

    The caller must treat False as "this frame was not written", because the audit row is the record that
    says a frame is clean. Writing the audit for an image that never reached storage would mark an
    unredacted frame as redacted, which is worse than the failure it is papering over.
    """
    for delay in (*_WRITE_RETRY_DELAYS_S, None):
        try:
            store.put_bytes(key, data, content_type)
            return True
        except Exception as exc:  # noqa: BLE001
            # Anything that is not the store shedding load is a real fault and belongs to the caller.
            if not _is_write_pressure(exc):
                raise
            if delay is None:
                log.warning("backfill.write_refused", key=key, error=str(exc))
                return False
            log.warning("backfill.write_deferred", key=key, delay_s=delay, error=str(exc))
            await asyncio.sleep(delay)
    return False


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
        # Overwrite the stored image at its own key. If the store will not take it, leave the frame exactly
        # as it was: no audit row, so it still matches the stale-method query and a later run picks it up.
        if not await _put_with_backoff(store, key, buf.tobytes(), "image/jpeg"):
            raise StoreRefused(key)
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
    totals = {"frames": 0, "cleaned": 0, "n_faces": 0, "n_plates": 0, "refused": 0}

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

    refused_in_a_row = 0
    for frame in frames:
        async with maker() as db:
            try:
                res = await backfill_frame(db, store, anonymizer, frame, redo=stale_method)
            except StoreRefused:
                # The frame keeps its old audit, so it is still selected by the next run. A store under
                # momentary pressure must not cost the 30,000 frames already done.
                totals["refused"] += 1
                refused_in_a_row += 1
                if refused_in_a_row >= _CONSECUTIVE_FAILURE_LIMIT:
                    log.error("backfill.store_unavailable", **totals)
                    break
                continue
            if res is None:
                continue
            await db.commit()
        refused_in_a_row = 0
        totals["frames"] += 1
        if res["n_faces"] or res["n_plates"]:
            totals["cleaned"] += 1
        totals["n_faces"] += res["n_faces"]
        totals["n_plates"] += res["n_plates"]

    log.info("backfill.done", **totals)
    return totals
