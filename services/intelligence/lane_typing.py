"""Type a session's lanes from the frames they were drawn on.

Mirrors `lane_linking`: a cheap pass over data that already exists, filling a column that has never held an
observation. Linking gave a lane an identity across frames; this gives it a kind, and the event layer needs
both. Without identity there is no crossing to detect, and without a kind every crossing detected is an
offence.

Images are the cost here, so lanes are grouped by frame and each frame is decoded once. A session with 674
lane frames and 1,853 lanes decodes 674 times rather than 1,853.

A lane a person typed is never overwritten. That is the same rule the reasoner's rerun and the event
derivation follow, for the same reason: a human decision is the output the loop exists to collect, and a
machine revisiting it undoes the work. Propagated lanes *are* reclassified, because their type was inherited
from whichever keyframe they came from rather than read off their own frame, and a lane that was solid where
it was drawn may be dashed two seconds later.
"""

from __future__ import annotations

import uuid as _uuid

from core.logging import get_logger

log = get_logger("lane_typing")


async def classify_session_lanes(db, session_id, *, apply: bool = True,
                                 reclassify: bool = False, limit: int | None = None) -> dict:
    """Read each lane's type off its frame.

    `reclassify` decides whether lanes that already carry a measured confidence are looked at again. Off by
    default so a re-run is cheap and idempotent; on when the classifier or its thresholds have changed and
    the previous answers are the thing being replaced.
    """
    from sqlalchemy import select

    from core.storage import get_object_store
    from db.models import Frame, Lane
    from services.autolabel.lane.linetype import MODEL_VERSION, classify_lane
    from services.recall.backends import load_image_bgr

    sid = session_id if isinstance(session_id, _uuid.UUID) else _uuid.UUID(str(session_id))
    q = (select(Lane, Frame.img_uri, Frame.width)
         .join(Frame, Lane.frame_id == Frame.frame_id)
         .where(Lane.session_id == sid, Lane.source != "human"))
    if not reclassify:
        q = q.where(Lane.marking_conf.is_(None))
    if limit:
        q = q.limit(limit)
    rows = (await db.execute(q)).all()
    if not rows:
        return {"session_id": str(sid), "lanes": 0,
                "detail": "no lanes to type: they are all human-typed or already measured"}

    by_frame: dict = {}
    for lane, img_uri, width in rows:
        by_frame.setdefault((str(lane.frame_id), img_uri, int(width or 0)), []).append(lane)

    store = get_object_store()
    counts: dict[str, int] = {}
    changed = 0
    unreadable = 0
    conf_sum = 0.0

    for (frame_id, img_uri, width), lanes in by_frame.items():
        try:
            img = load_image_bgr(store, img_uri)
        except Exception as exc:  # noqa: BLE001
            # One unreadable frame costs its own lanes, not the session. They keep a null confidence, which
            # is how the next run finds them again.
            log.warning("lane_typing.frame_unreadable", frame=frame_id, error=str(exc))
            unreadable += len(lanes)
            continue

        for lane in lanes:
            lane_type, conf, evidence = classify_lane(img, lane.control_points,
                                                      frame_width=width or None)
            counts[lane_type] = counts.get(lane_type, 0) + 1
            conf_sum += conf
            if not apply:
                continue
            if lane.lane_type != lane_type:
                changed += 1
            lane.lane_type = lane_type
            lane.marking_conf = conf
            # model_version records what proposed the geometry, and this pass proposed no geometry. Writing
            # the classifier's version here destroys the record of which detector drew the line, which is
            # exactly what happened on the first backfill: 4,554 lanes lost the proposer they came from.
            # The classifier identifies itself inside its own evidence instead.
            lane.provenance = {**(lane.provenance or {}),
                               "linetype": {**evidence.as_dict(), "model": MODEL_VERSION}}

    if apply:
        await db.commit()

    measured = sum(counts.values())
    log.info("lane_typing.done", session=str(sid), measured=measured, changed=changed,
             unreadable=unreadable, by_type=counts)
    return {"session_id": str(sid), "lanes": len(rows), "measured": measured,
            "changed_type": changed, "unreadable_frames_lanes": unreadable,
            "frames_decoded": len(by_frame),
            "by_type": dict(sorted(counts.items())),
            "mean_confidence": round(conf_sum / measured, 3) if measured else None,
            # The number that says whether this achieved anything. A session that comes back all one type
            # has been measured and found uniform, or has been measured badly, and the caller needs to be
            # able to tell those apart by looking rather than by trusting.
            "distinct_types": len(counts),
            "dry_run": not apply}


async def corpus_type_summary(db) -> dict:
    """What the corpus believes about its lane types, and how much of that was ever measured."""
    from sqlalchemy import func, select

    from db.models import Lane

    rows = (await db.execute(
        select(Lane.lane_type, func.count(), func.avg(Lane.marking_conf))
        .group_by(Lane.lane_type))).all()
    measured = (await db.execute(
        select(func.count()).select_from(Lane).where(Lane.marking_conf.isnot(None)))).scalar()
    total = (await db.execute(select(func.count()).select_from(Lane))).scalar()

    return {
        "total": int(total or 0),
        "measured": int(measured or 0),
        # Anything unmeasured is carrying the old hardcoded default, not an observation, and the difference
        # matters to every consumer that treats solid as a reason to call a crossing an offence.
        "unmeasured": int((total or 0) - (measured or 0)),
        "by_type": {str(t): {"count": int(n),
                             "mean_confidence": round(float(c), 3) if c is not None else None}
                    for t, n, c in rows},
    }
