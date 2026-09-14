"""Whether a session can support cross-view linking at all, and exactly what is missing when it cannot.

The cross-camera machinery here is complete. `services/agent/crosscam_agent.py::plan_cross_camera` returns
a candidate box and a visibility grade per camera; `commit_cross_camera` writes them as one reversible
run; `services/multicam/propagate.py` does the lens-aware ground-plane projection and already refuses with
`gated=True` when a session has not passed calibration validation.

What has been missing is a way to ask the question before opening the page. Measured over the corpus:

    sessions with more than one camera        6 of 377
    frame groups holding more than one camera   359 of 20,119   (1.8%)
    RigObject rows                              0
    the only five-camera session                1,928 frames and ZERO labelled objects
    camera_calibration rows                     101, every one source='estimated'

So the feature is not blocked on engineering, and an annotator opening the multi-camera page sees an inert
grid with no explanation. That is the worst state to be in: a working tool that looks broken. This module
turns the absence into a sentence, and orders the sentences by what has to happen first, because labelling
objects on a session whose calibration has never been validated fixes nothing.

Nothing here is a workaround. Cross-view linking genuinely requires a rig, a validated calibration and
something to link, and the honest thing to do about a missing precondition is name it.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import CalibrationValidation, Frame, FrameGroup, Object

log = get_logger("multicam.readiness")

# Below this a "rig" is one camera and there is no second view to link into.
MIN_CAMERAS = 2

# A session with a handful of synchronized groups can be linked by hand; below this there is not enough
# overlap for the feature to be worth opening.
MIN_GROUPS = 5


async def session_readiness(db: AsyncSession, session_id: UUID) -> dict:
    """What this session has, what it lacks, and what to do about it.

    Returns `ready` plus an ordered list of blockers. Ordered because the fixes depend on each other:
    labelling objects on a session whose calibration has never been validated produces labels that cannot
    be projected anywhere.
    """
    cams = [c for c in (await db.execute(
        select(Frame.cam_id).where(Frame.session_id == session_id).distinct())).scalars().all() if c]

    n_groups = (await db.execute(
        select(func.count()).select_from(FrameGroup)
        .where(FrameGroup.session_id == session_id))).scalar_one()

    # Groups that actually hold more than one camera. A group of one is a frame with a group id, and
    # counting those as rig coverage is how 20,119 groups look like a multi-camera corpus when 359 are.
    multi_groups = 0
    for row in (await db.execute(
            select(FrameGroup.frame_ids).where(FrameGroup.session_id == session_id))).scalars().all():
        if isinstance(row, dict) and len(row) > 1:
            multi_groups += 1

    validations = (await db.execute(
        select(CalibrationValidation).where(
            CalibrationValidation.session_id == session_id))).scalars().all()
    passed = {v.cam_id for v in validations if v.status == "pass"}

    n_objects = (await db.execute(
        select(func.count()).select_from(Object).join(Frame, Frame.frame_id == Object.frame_id)
        .where(Frame.session_id == session_id, Object.state != "rejected"))).scalar_one()

    blockers: list[dict] = []

    if len(cams) < MIN_CAMERAS:
        blockers.append({
            "code": "single_camera",
            "detail": f"this session has {len(cams) or 'no'} camera"
                      f"{'' if len(cams) == 1 else 's'}, so there is no second view to link into",
            "fix": "cross-view linking needs a rig; this is a capture-side fact and nothing here changes it",
        })

    if len(cams) >= MIN_CAMERAS and multi_groups < MIN_GROUPS:
        blockers.append({
            "code": "no_synchronized_groups",
            "detail": f"{multi_groups} of {n_groups} frame groups hold more than one camera, so the "
                      "cameras were rarely capturing at the same moment",
            "fix": "rebuild the frame groups for this session, or widen the synchronisation tolerance",
        })

    missing_cal = [c for c in cams if c not in passed]
    if missing_cal:
        blockers.append({
            "code": "calibration_not_validated",
            "detail": f"{len(missing_cal)} of {len(cams)} cameras have not passed calibration validation: "
                      + ", ".join(sorted(missing_cal)[:6]),
            # Not advice invented here: services/multicam/propagate.py refuses outright without it, because
            # projecting a box between cameras needs real intrinsics and extrinsics.
            "fix": "run calibration validation on this session; projection between cameras is gated on it",
        })

    if n_objects == 0:
        blockers.append({
            "code": "nothing_to_link",
            "detail": "this session has no labelled objects, so there is nothing to project into the "
                      "other views",
            "fix": "label some objects in one camera first; cross-view linking carries existing work "
                   "across, it does not create it",
        })

    return {
        "session_id": str(session_id),
        "ready": not blockers,
        "cameras": sorted(cams),
        "frame_groups": int(n_groups),
        "multi_camera_groups": multi_groups,
        "calibration_passed": sorted(passed),
        "objects": int(n_objects),
        # Ordered by what has to happen first. A caller showing only the head of this list is showing the
        # right one.
        "blockers": blockers,
    }
