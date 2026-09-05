"""Getting a 3D quality flag in front of somebody.

`services/lidar/quality3d/checker.py` finds real problems: a cuboid floating above the road, one below it,
impossible dimensions, a duplicate, a box whose reprojection does not match the 2D label it is linked to,
a cluster of points with no cuboid on it. It writes them to `quality_flag_3d`, which has its own table,
its own review endpoint and no reader anywhere an annotator goes. The 2D work has two queues people
actually work, `error_candidate` and `issue`, and neither has ever received a 3D flag.

So this is a bridge and deliberately nothing more. It does not re-derive anything.

**What it can and cannot carry across.** An `ErrorCandidate` is keyed on a 2D `Object`, so only a flag
whose cuboid is linked to one can become a candidate: `Object3D.object_id` is the unifying identity across
a box, a mask and a cuboid, and it is null on 42 of the corpus's 56 cuboids. A flag with no 2D object is
not silently dropped, it is reported as unbridgeable with the reason, because "no candidates were created"
and "these flags have nowhere to go" are different facts and only the second tells somebody what to fix.

**Nothing here is auto-confirmed.** Every candidate is `pending`, matching the 2D detectors, and the
evidence the checker recorded travels with it. `checker.py` notes that all 101 camera calibrations are
`source='estimated'` and that its own reprojection test is deliberately conservative because of it; a flag
built on an estimated extrinsic is a question, not a verdict.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import ErrorCandidate, Object3D, QualityFlag3D

log = get_logger("quality3d.bridge")

# What each 3D flag becomes in the 2D queue. Prefixed so a consumer can tell at a glance that the evidence
# came from the point cloud and not from the image, which matters: a 2D reviewer looking at a crop cannot
# see why a cuboid is floating.
KIND_PREFIX = "lidar_"

# Flags worth a person's time. `missing_neighbour` is deliberately absent: it names a cluster with no
# cuboid, so there is no object to hang a candidate on, and it belongs in a recall queue rather than an
# error queue.
BRIDGEABLE = ("floating", "below_ground", "impossible_dims", "duplicate", "misaligned",
              "box_2d3d_inconsistent")


async def bridge_flags(db: AsyncSession, *, cloud_id: UUID | None = None,
                       session_id: UUID | None = None, commit: bool = False) -> dict:
    """Turn open 3D quality flags into error candidates on the 2D objects they belong to.

    Returns what it did and, as importantly, what it could not do. Idempotent per (object, kind): a flag
    already bridged is not bridged again, so this is safe to run after every consistency pass.
    """
    q = (select(QualityFlag3D, Object3D)
         .join(Object3D, Object3D.object_3d_id == QualityFlag3D.object_3d_id)
         .where(QualityFlag3D.status == "open"))
    if cloud_id is not None:
        q = q.where(QualityFlag3D.cloud_id == cloud_id)
    if session_id is not None:
        from db.models import PointCloud

        q = q.join(PointCloud, PointCloud.cloud_id == Object3D.cloud_id).where(
            PointCloud.session_id == session_id)
    rows = (await db.execute(q)).all()

    if not rows:
        return {"created": 0, "unbridgeable": [], "flags": 0,
                "reason": "no open 3D quality flags in scope; run the consistency pass first"}

    # What is already there, so a re-run adds nothing.
    linked_ids = [o3d.object_id for _f, o3d in rows if o3d.object_id is not None]
    existing = set()
    if linked_ids:
        existing = {(str(c.object_id), c.kind) for c in (await db.execute(
            select(ErrorCandidate).where(ErrorCandidate.object_id.in_(linked_ids)))).scalars().all()}

    created, unbridgeable, skipped = 0, [], 0
    for flag, o3d in rows:
        if flag.kind not in BRIDGEABLE:
            unbridgeable.append({"flag_id": str(flag.flag_id), "kind": flag.kind,
                                 "reason": "this flag names a place with no cuboid, so there is no 2D "
                                           "object to raise it against"})
            continue
        if o3d.object_id is None:
            # The common case today and the useful thing to report: 42 of 56 cuboids in this corpus have
            # no 2D object behind them, so most 3D findings have nowhere to surface.
            unbridgeable.append({"flag_id": str(flag.flag_id), "kind": flag.kind,
                                 "object_3d_id": str(flag.object_3d_id),
                                 "reason": "this cuboid is not linked to a 2D object, so there is no "
                                           "queue entry to create; link it first"})
            continue
        kind = f"{KIND_PREFIX}{flag.kind}"
        if (str(o3d.object_id), kind) in existing:
            skipped += 1
            continue
        db.add(ErrorCandidate(
            object_id=o3d.object_id, kind=kind, score=float(flag.score or 0.0),
            # No proposed label. A floating cuboid is a geometry problem and the class is not in question,
            # and a proposal the detector cannot justify is worse than none: the 2D queue binds a key to
            # applying it.
            proposed_label=None,
            detail={**(flag.detail or {}), "flag_id": str(flag.flag_id),
                    "object_3d_id": str(flag.object_3d_id),
                    "source": "lidar_quality3d",
                    # Carried so a reviewer looking at a crop knows why a claim about depth is being made
                    # about it at all.
                    "note": "found in the point cloud, not in this image"},
            status="pending"))
        existing.add((str(o3d.object_id), kind))
        created += 1

    if commit:
        await db.commit()
    log.info("quality3d.bridged", created=created, skipped=skipped, unbridgeable=len(unbridgeable))
    return {"created": created, "already_present": skipped, "flags": len(rows),
            "unbridgeable": unbridgeable}
