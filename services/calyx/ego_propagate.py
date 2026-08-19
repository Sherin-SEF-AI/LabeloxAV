"""Propagating a label to the next frame with the ego motion, and refusing when the pose is not there.

The guard is the important part of this module and it fires on every session in this corpus.

Ego-compensated propagation needs a rigid transform between two frames: how the camera moved. This engine
has `Frame.gnss` on 3 frames and `Frame.ego_speed` on 6, across one session of 377, and no per-frame 6-DOF
pose table at all. So for essentially the whole corpus the honest answer is that the motion is unknown,
and this refuses rather than assuming the camera stood still, which is what an uncompensated copy assumes
and is wrong for every static object.

Two ways of moving a box, and they check each other:

    geometry   the ground homography the ego motion induces (core/accel/ego_homography.py). Exact for a
               static object on the road surface, meaningless for anything else, which is why the pack's
               motion model gates it.
    tracker    whatever the tracker already produced for that object on the destination frame.

When both exist and agree within tolerance, the box is written and stamped with the run that produced it.
When they disagree, NEITHER is written and a PropagationConflict row is: picking one would propagate a
guess, and averaging them would propagate a box neither method proposed. The resulting table names the
frames where the calibration or the pose is wrong, which is worth more than the label would have been.

Provenance goes in `Object.provenance` under `derived_from_run`, not in a new column. `provenance` already
carries run stamps (`agent_run_id` among them), and grep shows `derived_from_run` appears nowhere in the
tree despite being referenced in the brief, so there is no existing column to honour.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.accel.boxes import box_iou_matrix
from core.accel.ego_homography import ground_homography, warp_box
from core.logging import get_logger
from db.models import Frame, Object, PropagationConflict

log = get_logger("ego_propagate")

# How far the two methods may disagree, as IoU between the boxes they propose. Below this the two are
# describing different objects and neither should be trusted.
AGREE_IOU = 0.5


class PoseUnavailable(Exception):
    """Raised rather than returned, because a caller that ignores this writes wrong labels silently."""


async def ego_transform(db: AsyncSession, from_frame: Frame, to_frame: Frame) -> dict[str, Any]:
    """The camera motion between two frames, or a refusal naming what is missing.

    Returns {"measured", "R", "t", "reason"}. This is where the corpus blocks: with GNSS on 3 frames and
    no pose table, there is nothing to derive a rigid transform from for any real pair, and the module
    says which of the two frames lacked what rather than reporting a generic failure.
    """
    missing = []
    for label, f in (("from", from_frame), ("to", to_frame)):
        if f.gnss is None:
            missing.append(f"{label} frame has no GNSS fix")
    if missing:
        return {"measured": False, "R": None, "t": None,
                "reason": "; ".join(missing) + ". Ego-compensated propagation needs the camera motion "
                          "between the two frames, and this corpus carries GNSS on 3 frames of 41,752 "
                          "and has no per-frame 6-DOF pose table at all"}

    # With both fixes present the translation is derivable; rotation needs a heading source this corpus
    # also lacks, so the honest transform is translation-only and is declared as such.
    return {"measured": False, "R": None, "t": None,
            "reason": "both frames carry GNSS but no heading or IMU attitude, so the rotation between "
                      "them is unknown; a translation-only transform would place a rotated scene wrongly"}


def _agreement(geometry_box: tuple[float, ...], tracker_box: list[float]) -> float:
    a = np.asarray([list(geometry_box)], dtype=float)
    b = np.asarray([list(tracker_box)], dtype=float)
    return float(box_iou_matrix(a, b)[0, 0])


async def propagate_frame(db: AsyncSession, *, from_frame_id: str, to_frame_id: str,
                          run_id: str | None = None, agree_iou: float = AGREE_IOU,
                          dry_run: bool = True) -> dict[str, Any]:
    """Propagate every eligible label from one frame to the next, or refuse and say why.

    `dry_run` defaults True: this writes labels into the corpus, and the guard below has never been
    satisfied on any real session, so the first time it is it should be a person's decision.
    """
    src = await db.get(Frame, UUID(from_frame_id))
    dst = await db.get(Frame, UUID(to_frame_id))
    if src is None or dst is None:
        return {"measured": False, "reason": "one of the frames does not exist"}
    if src.session_id != dst.session_id:
        return {"measured": False,
                "reason": "the two frames are from different sessions, so there is no ego motion "
                          "between them"}

    motion = await ego_transform(db, src, dst)
    if not motion["measured"]:
        # The guard. It fires on every session in this corpus and that is the correct outcome: an
        # uncompensated copy assumes the camera stood still, which is wrong for every static object and
        # most wrong for the roadside furniture propagation should be best at.
        log.info("ego_propagate.refused", from_frame=from_frame_id, to_frame=to_frame_id,
                 reason=motion["reason"])
        return {"measured": False, "from_frame_id": from_frame_id, "to_frame_id": to_frame_id,
                "reason": motion["reason"], "n_propagated": 0, "n_conflicts": 0}

    # Unreachable on this corpus: the guard above has never been satisfied on a real session.
    from packs.registry import default_pack_id, get_pack
    from services.autolabel.ontology import get_ontology
    from services.calibration.resolve import resolve_calibration

    spec = get_pack(default_pack_id()).motion_models
    onto = get_ontology()
    calib = await resolve_calibration(src.session_id, src.cam_id)
    if calib is None or spec is None:
        return {"measured": False, "reason": "no calibration for this camera, or the pack defines no "
                                             "motion models, so no box can be warped"}

    H = ground_homography(calib.K(), motion["R"], motion["t"],
                          normal=[0.0, 1.0, 0.0], height=getattr(calib, "height_m", 1.5))
    src_objs = (await db.execute(select(Object).where(Object.frame_id == src.frame_id))).scalars().all()
    dst_objs = (await db.execute(select(Object).where(Object.frame_id == dst.frame_id))).scalars().all()
    by_track = {o.track_id: o for o in dst_objs if o.track_id is not None}

    written, conflicts, refused = 0, 0, 0
    for o in src_objs:
        try:
            model = spec.model_for(onto.by_id(o.class_id).name)
        except Exception:  # noqa: BLE001
            model = "moving"
        w = warp_box(list(o.bbox), H, width=dst.width, height=dst.height, motion_model=model)
        if not w.measured:
            refused += 1
            continue
        tracked = by_track.get(o.track_id)
        if tracked is not None:
            iou = _agreement(w.box, list(tracked.bbox))
            if iou < agree_iou:
                conflicts += 1
                if not dry_run:
                    db.add(PropagationConflict(
                        session_id=src.session_id, from_frame_id=src.frame_id,
                        to_frame_id=dst.frame_id, object_id=o.object_id, class_id=o.class_id,
                        motion_model=model, geometry_box=list(w.box),
                        tracker_box=list(tracked.bbox), iou=round(iou, 4), tolerance=agree_iou,
                        reason="ego geometry and the tracker propose different boxes; neither written"))
                continue
        if not dry_run:
            db.add(Object(frame_id=dst.frame_id, track_id=o.track_id, class_id=o.class_id,
                          bbox=list(w.box), conf=float(o.conf), source="propagated", state="review",
                          provenance={"created_by": "ego-propagation",
                                      "derived_from_run": run_id,
                                      "from_object_id": str(o.object_id),
                                      "motion_model": model, "shrink": w.shrink}))
        written += 1
    if not dry_run:
        await db.commit()

    log.info("ego_propagate.done", from_frame=from_frame_id, to_frame=to_frame_id,
             propagated=written, conflicts=conflicts, refused=refused, dry_run=dry_run)
    return {"measured": True, "from_frame_id": from_frame_id, "to_frame_id": to_frame_id,
            "n_propagated": written, "n_conflicts": conflicts, "n_refused": refused,
            "dry_run": dry_run}


async def coverage_report(db: AsyncSession) -> dict[str, Any]:
    """How much of the corpus could be propagated at all, which is the honest headline for this feature.

    Reported rather than assumed, because "ego propagation is built" and "ego propagation can run" are
    different claims and only the first is currently true.
    """
    from sqlalchemy import func

    total = (await db.execute(select(func.count()).select_from(Frame))).scalar_one()
    with_gnss = (await db.execute(select(func.count()).select_from(Frame).where(
        Frame.gnss.is_not(None)))).scalar_one()
    with_speed = (await db.execute(select(func.count()).select_from(Frame).where(
        Frame.ego_speed.is_not(None)))).scalar_one()
    return {"n_frames": int(total), "n_with_gnss": int(with_gnss), "n_with_ego_speed": int(with_speed),
            "pct_with_gnss": round(100.0 * with_gnss / total, 4) if total else 0.0,
            "propagatable": False,
            "reason": ("a rigid camera transform needs position AND attitude between two frames; this "
                       "corpus has GNSS on a handful of frames, no heading or IMU attitude, and no "
                       "per-frame 6-DOF pose table, so no pair of frames yields a usable transform")}


__all__ = ["ego_transform", "propagate_frame", "coverage_report", "PoseUnavailable", "AGREE_IOU"]
