"""Proposing one frame's lanes and writing them, shared by the editor's button and the corpus runner.

It lived inside the HTTP route, which made the route the only way lanes were ever proposed: a corpus pass
had nothing to call. The rules are unchanged and are recorded where they are applied below.
"""

from __future__ import annotations

import json

import numpy as np
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, Lane
from services.autolabel.lane.curves import fit_control_points, mark_ego
from services.autolabel.lane.detect import model_tag, propose_lanes
from services.autolabel.lane.linetype import MODEL_VERSION as LINETYPE_VERSION
from services.autolabel.lane.linetype import classify_lane
from services.autolabel.lane.plausible import filter_proposals

log = get_logger("lanes.propose")


async def drivable_classes(db: AsyncSession, frame_id) -> dict | None:
    """The frame's drivable polygons, or None if nobody has segmented it.

    None and an empty mask are different and the plausibility test treats them differently: no mask is
    missing evidence, not evidence that the road is missing.
    """
    from db.models import DrivableMask

    dm = await db.get(DrivableMask, frame_id)
    if dm is None or not dm.mask_uri:
        return None
    try:
        return json.loads(get_object_store().get_bytes(dm.mask_uri)).get("classes") or None
    except Exception as exc:  # noqa: BLE001
        log.warning("lanes.drivable_unreadable", frame=str(frame_id), error=str(exc))
        return None


async def propose_frame_lanes(db: AsyncSession, frame: Frame, img: np.ndarray) -> tuple[list[Lane], list]:
    """Replace the frame's machine-proposed lanes with fresh ones; returns (created, rejected evidence).

    Only `source='proposed'` rows are replaced, so a lane a person drew or edited is never touched. Flushes
    but does not commit; the caller owns the transaction.
    """
    await db.execute(delete(Lane).where(Lane.frame_id == frame.frame_id, Lane.source == "proposed"))
    cps = [fit_control_points(p) for p in propose_lanes(img)]

    # A lane sits on the road. The detector finds bright linear structure, and a dashcam frame is full of
    # bright linear structure that is not a lane: a striped hoarding, a flyover girder, a kerb, an awning.
    # On one real frame that produced six "lanes", every one an edge of a striped wall, drawn as diagonals
    # across the sky. The drivable mask already knows where the road is, so it is asked.
    surface = await drivable_classes(db, frame.frame_id)
    kept, rejected = filter_proposals(cps, surface, frame.width or 0)
    if rejected:
        log.info("lanes.rejected_off_surface", frame=str(frame.frame_id), n=len(rejected), kept=len(kept))
    cps = [c for c, _ev in kept]

    ego = mark_ego(cps, frame.width, frame.height)
    created = []
    for i, cp in enumerate(cps):
        # Typed from the paint on the way in. This used to be the literal "solid" for every lane the system
        # had ever proposed, which is what left 4,548 of 4,558 lanes claiming a type nobody measured.
        lane_type, conf, evidence = classify_lane(img, cp, frame_width=frame.width)
        lane = Lane(frame_id=frame.frame_id, session_id=frame.session_id, control_points=cp,
                    lane_type=lane_type, is_ego=(i == ego), source="proposed",
                    marking_conf=conf, provenance={"linetype": evidence.as_dict()},
                    model_version=f"{model_tag()}+{LINETYPE_VERSION}")
        db.add(lane)
        created.append(lane)
    await db.flush()
    return created, [ev for _c, ev in rejected]
