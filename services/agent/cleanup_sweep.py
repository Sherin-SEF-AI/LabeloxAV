"""Cleanup sweep: apply the new detection gates to objects ALREADY in the corpus, without re-running any
model. It removes exactly the errors the panoptic/quality fixes prevent going forward:

  - stuff instances   (a boxed tree, barrier, wall, building, sky -> belongs to semantic seg, not a box)
  - ego-hood boxes    (a detection mostly inside the camera's estimated hood mask -> the car labeling itself)
  - oversize boxes    (a single instance spanning most of the frame -> a fusion "everything" artifact)
  - duplicate boxes   (the same object wrapped in several overlapping/nested boxes -> keep the best one)

It never touches a human-labelled object. Every removed object is snapshotted into the run, so the whole
sweep reverts by re-inserting them. Fast (DB + geometry only), so it fixes the visible mess in minutes
instead of the ~12-day GPU cost of a full re-detection.
"""

from __future__ import annotations

import uuid

from sqlalchemy import distinct, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Frame, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker

log = get_logger("agent.cleanup")

_MACHINE = ("fused", "auto_accept", "interpolated")   # machine-written sources; never "human" or "imported"
# columns to snapshot so a removed object re-inserts faithfully on revert (created_at/version regenerate)
_SNAP_COLS = ("object_id", "frame_id", "track_id", "class_id", "bbox", "mask_uri", "mask_encoding", "attrs",
              "conf", "source", "provenance", "state", "cuboid_3d", "rot_deg", "keypoints", "polyline",
              "is_keyframe", "interp_source", "sign_type", "sign_category", "ocr_text", "ocr_lang", "ocr_conf")


def _snapshot(o: Object) -> dict:
    snap = {}
    for c in _SNAP_COLS:
        v = getattr(o, c)
        snap[c] = str(v) if isinstance(v, uuid.UUID) else v
    return snap


def _restore(snap: dict) -> Object:
    kw = dict(snap)
    for c in ("object_id", "frame_id", "track_id"):
        if kw.get(c):
            kw[c] = uuid.UUID(kw[c])
    return Object(**kw)


def _dup_removals(objs: list[Object], onto) -> set:
    """Object ids to drop as duplicates: for each physical object keep the highest-confidence box. Same
    object = same class / same l1 superclass / a fallback, overlapping by IoU or nested (IoM). Mirrors the
    fusion de-dup so existing data matches what new detection would now produce."""
    from core.config import get_settings
    from services.autolabel.fusion import _iom, _iou

    cfg = get_settings().fusion
    kept: list[Object] = []
    drop: set = set()
    for o in sorted(objs, key=lambda x: float(x.conf or 0), reverse=True):
        box = tuple(float(v) for v in o.bbox)
        is_dup = False
        for k in kept:
            same = (o.class_id == k.class_id or onto.is_fallback(o.class_id) or onto.is_fallback(k.class_id)
                    or onto.by_id(o.class_id).l1 == onto.by_id(k.class_id).l1)
            if not same:
                continue
            kbox = tuple(float(v) for v in k.bbox)
            if _iou(box, kbox) >= cfg.dedupe_iou or _iom(box, kbox) >= cfg.dedupe_iom:
                is_dup = True
                break
        if is_dup:
            drop.add(o.object_id)
        else:
            kept.append(o)
    return drop


def _reason(o: Object, onto, frame_w: int, frame_h: int, ego, max_area_frac: float) -> str | None:
    if onto.is_stuff(o.class_id):
        return "stuff"
    x1, y1, x2, y2 = (float(v) for v in o.bbox)
    if (max(1.0, (x2 - x1)) * max(1.0, (y2 - y1))) / max(1.0, float(frame_w) * float(frame_h)) > max_area_frac:
        return "oversize"
    if ego is not None and ego.contains_bbox((x1, y1, x2, y2), frame_w, frame_h):
        return "ego_hood"
    return None


async def _sweep_frame(db: AsyncSession, frame: Frame, vehicle_id: str | None, onto, max_area_frac: float,
                       run_id: uuid.UUID) -> tuple[list[dict], dict[str, int]]:
    from services.autolabel.ego_mask import get_ego_mask

    objs = (await db.execute(select(Object).where(
        Object.frame_id == frame.frame_id, Object.source.in_(_MACHINE)))).scalars().all()
    if not objs:
        return [], {}
    ego = get_ego_mask(vehicle_id, frame.cam_id) if vehicle_id else None
    counts = {"stuff": 0, "oversize": 0, "ego_hood": 0, "duplicate": 0}

    to_remove: dict = {}
    for o in objs:
        r = _reason(o, onto, frame.width, frame.height, ego, max_area_frac)
        if r:
            to_remove[o.object_id] = r
    # de-dup only among the objects that survive the reason filter (do not double-count a removed stuff box)
    survivors = [o for o in objs if o.object_id not in to_remove]
    for oid in _dup_removals(survivors, onto):
        to_remove[oid] = "duplicate"

    snaps = []
    for o in objs:
        if o.object_id in to_remove:
            snaps.append({**_snapshot(o), "_reason": to_remove[o.object_id]})
            counts[to_remove[o.object_id]] = counts.get(to_remove[o.object_id], 0) + 1
            await db.delete(o)
    return snaps, counts


async def run_cleanup_sweep(run_id: uuid.UUID, *, do_pii: bool = True, pii_limit: int = 5000) -> None:
    """Background: sweep every frame with machine objects, remove the four error classes, snapshot removals
    for revert, and optionally backfill PII on pre-gate frames. Updates the AgentRun as it goes."""
    from core.config import get_settings
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    max_area = get_settings().quality.max_area_frac
    maker = get_sessionmaker()

    async with maker() as db:
        veh_rows = (await db.execute(select(DbSession.session_id, DbSession.vehicle_id))).all()
        vehicle_by_session = {str(sid): vid for sid, vid in veh_rows}
        frame_ids = list((await db.execute(
            select(distinct(Object.frame_id)).where(Object.source.in_(_MACHINE)))).scalars().all())

    totals = {"frames": 0, "removed": 0, "stuff": 0, "oversize": 0, "ego_hood": 0, "duplicate": 0}
    removed_snaps: list[dict] = []
    try:
        for fid in frame_ids:
            async with maker() as db:
                frame = await db.get(Frame, fid)
                if frame is None:
                    continue
                vid = vehicle_by_session.get(str(frame.session_id))
                snaps, counts = await _sweep_frame(db, frame, vid, onto, max_area, run_id)
                if snaps:
                    await db.commit()
                else:
                    continue
            totals["frames"] += 1
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
            totals["removed"] += len(snaps)
            removed_snaps.extend(snaps)
            if totals["frames"] % 200 == 0:
                async with maker() as db:
                    run = await db.get(AgentRun, run_id)
                    if run:
                        run.counts = dict(totals)
                        await db.commit()

        pii_result = None
        if do_pii:
            try:
                from services.anonymize.backfill import backfill_unaudited

                pii_result = await backfill_unaudited(limit=pii_limit)
            except Exception as exc:  # noqa: BLE001 - PII is best-effort here; the box cleanup already committed
                log.warning("cleanup.pii_failed", error=str(exc))
                pii_result = {"error": str(exc)}

        async with maker() as db:
            run = await db.get(AgentRun, run_id)
            if run:
                run.status = "committed"
                run.counts = {**totals, "pii": pii_result}
                run.changes = {"removed": removed_snaps}
                await db.commit()
        log.info("cleanup.done", **totals)
    except Exception as exc:  # noqa: BLE001
        log.error("cleanup.failed", run_id=str(run_id), error=str(exc))
        async with maker() as db:
            run = await db.get(AgentRun, run_id)
            if run:
                run.status, run.error = "error", str(exc)
                run.changes = {"removed": removed_snaps}
                await db.commit()


async def revert_cleanup(db: AsyncSession, run: AgentRun) -> dict:
    """Re-insert every object the sweep removed (skipping any id that already exists again)."""
    restored = 0
    for snap in (run.changes or {}).get("removed", []):
        oid = uuid.UUID(snap["object_id"])
        if await db.get(Object, oid) is not None:
            continue
        clean = {k: v for k, v in snap.items() if not k.startswith("_")}
        db.add(_restore(clean))
        restored += 1
    run.status = "reverted"
    from datetime import datetime, timezone
    run.reverted_at = datetime.now(timezone.utc)
    await db.commit()
    log.info("cleanup.revert", run_id=str(run.run_id), restored=restored)
    return {"run_id": str(run.run_id), "restored": restored}
