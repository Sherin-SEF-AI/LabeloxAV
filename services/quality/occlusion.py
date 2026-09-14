"""Which object is in front, from depth that was already being computed.

Both halves of this have existed and were never joined. `ObjectRelationship` has allowed
`kind="occludes"` with `source="geometry"` since relations were added, and nothing has ever written one:
zero rows in the corpus. `ObjectDynamics.distance_m` holds a ground-contact depth for 367,000 objects
across 33,767 frames, computed by lifting each box's bottom centre through the same IPM ground plane the
cuboid solve uses. Two boxes that overlap in the image and differ in depth have an occlusion order, and
nothing was asking.

**How much this can actually resolve, measured over 300 frames and 32,592 object pairs:**

    overlapping by at least 15% of the smaller box   2,843   (8.7% of pairs)
    depth separates them by more than their error      966   (34%)
    too close in depth to order                      1,877   (66%)

966 occlusion relations from 300 frames, where the corpus held zero. The 66% are not guessed: two objects
whose depths differ by less than those depths' own error genuinely have no order this method can see, and
inventing one would be worse than leaving it open, because a wrong occlusion order is invisible in the
label and wrong in every consumer that reads it.

A first version of that threshold was a flat 40 m cutoff and it ordered ZERO pairs on the first three real
frames tried, because the overlapping objects in this corpus sit at 58, 76 and 99 m. The rule that works
is the one the stored depth already obeys, scaled with the estimate rather than picked.

**The depth is monocular and approximate.** `services/dynamics/compute.py` records that plainly: the
method is `ipm_mono_v1`, it assumes a flat ground plane, and its own documentation puts a 79.6% phantom
U-turn rate on trajectories built from it. So every relation this writes is `status="proposed"`, never
confirmed, and carries the depth gap it was derived from, so a reviewer can see how much daylight there
was.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Object, ObjectDynamics, ObjectRelationship

log = get_logger("occlusion")

# How much of the SMALLER box must be covered before the pair is treated as overlapping.
#
# The smaller box rather than the union, because a scooter in front of a bus covers a tiny fraction of the
# bus and all of itself, and that is exactly the occlusion worth recording.
MIN_OVERLAP_FRAC = 0.15

# How far apart in depth two objects must be before one is called nearer.
#
# An absolute floor, for the near field where a metre is genuinely below the noise of a ground-plane lift.
MIN_DEPTH_GAP_M = 1.0

# The gap must also clear the two estimates' own error, which grows with distance.
#
# `services/dynamics/compute.py` derives the error of a flat-road IPM lift as df/f = f / (fy * h), and only
# stores a distance while that stays inside IPM_ERROR_BUDGET, 25%. So a stored depth carries up to a
# quarter of itself in error by construction, and two depths are only distinguishable when they differ by
# more than the sum of their errors.
#
# A flat cutoff was tried first and was wrong in the way invented constants usually are. Capping at 40 m
# ordered ZERO pairs on three real frames, because the overlapping objects in this corpus sit at 58, 76
# and 99 m - well inside what the codebase's own bound already permits, and rejected only by a number
# picked out of the air. Scaling with the estimate makes the rule the same one the stored value already
# obeys.
DEPTH_ERROR_FRAC = 0.25


def overlap_fraction(a: list[float], b: list[float]) -> float:
    """Intersection as a fraction of the smaller box's area."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / min(aa, bb))


def _confidence(gap_m: float, near_conf: float, far_conf: float) -> float:
    """How much to trust this ordering.

    Built from the two depth estimates' own confidences and from how far apart they are: a two-metre gap
    between two poorly-estimated objects is a weaker claim than a twenty-metre gap between two good ones.
    """
    gap_term = min(1.0, gap_m / 10.0)
    return round(min(0.95, 0.35 + 0.4 * gap_term + 0.2 * min(near_conf, far_conf)), 3)


async def propose_occlusion(db: AsyncSession, frame_id: UUID) -> dict:
    """Occlusion order for one frame, as proposed relations plus the pairs it could not order.

    Returns what it would write. Nothing is committed here; `commit_occlusion` does that, so the caller
    can look first. Pairs that overlap and cannot be ordered are returned too, because "these two overlap
    and nobody knows which is in front" is a real answer and the only one that leads to a person looking.
    """
    rows = (await db.execute(
        select(Object.object_id, Object.bbox, Object.class_id,
               ObjectDynamics.distance_m, ObjectDynamics.confidence)
        .join(ObjectDynamics, ObjectDynamics.object_id == Object.object_id)
        .where(Object.frame_id == frame_id, Object.state != "rejected",
               ObjectDynamics.distance_m.is_not(None)))).all()

    n_objects = (await db.execute(
        select(Object.object_id).where(Object.frame_id == frame_id, Object.state != "rejected"))).all()

    if len(rows) < 2:
        return {"frame_id": str(frame_id), "pairs": [], "unordered": [], "n_with_depth": len(rows),
                "n_objects": len(n_objects),
                "reason": "fewer than two objects on this frame carry a depth estimate; "
                          "run the dynamics pass over the session first"}

    pairs, unordered = [], []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            frac = overlap_fraction([float(v) for v in a.bbox], [float(v) for v in b.bbox])
            if frac < MIN_OVERLAP_FRAC:
                continue
            da, dbb = float(a.distance_m), float(b.distance_m)
            gap = abs(da - dbb)
            far_depth = max(da, dbb)
            # The gap has to clear both the near-field floor and the two estimates' own error.
            needed = max(MIN_DEPTH_GAP_M, DEPTH_ERROR_FRAC * (da + dbb))
            if gap < needed:
                unordered.append({"a": str(a.object_id), "b": str(b.object_id),
                                  "overlap": round(frac, 3), "depth_gap_m": round(gap, 2),
                                  "needed_gap_m": round(needed, 2),
                                  "reason": f"they are {gap:.1f} m apart at {da:.0f} and {far_depth:.0f} m, "
                                            f"and a monocular lift needs {needed:.1f} m at that range "
                                            "before one can be called nearer"})
                continue
            near, far = (a, b) if da < dbb else (b, a)
            pairs.append({
                "from_object_id": str(near.object_id), "to_object_id": str(far.object_id),
                "overlap": round(frac, 3), "depth_gap_m": round(gap, 2),
                "near_m": round(min(da, dbb), 2), "far_m": round(far_depth, 2),
                "conf": _confidence(gap, float(near.confidence or 0.5), float(far.confidence or 0.5)),
            })

    return {"frame_id": str(frame_id), "pairs": pairs, "unordered": unordered,
            "n_with_depth": len(rows), "n_objects": len(n_objects)}


async def commit_occlusion(db: AsyncSession, frame_id: UUID) -> dict:
    """Write the proposed occlusion order as `occludes` relations.

    Every row is `status="proposed"` and `source="geometry"`, never confirmed. The depth behind it is a
    monocular ground-plane estimate, and a relation that claimed to be confirmed would be asserting a fact
    about the world on the strength of an approximation.

    Idempotent per ordered pair: re-running on an unchanged frame writes nothing, so this is safe to call
    after every dynamics pass.
    """
    plan = await propose_occlusion(db, frame_id)
    if not plan["pairs"]:
        return {**plan, "created": 0}

    existing = {(str(r.from_object_id), str(r.to_object_id)) for r in (await db.execute(
        select(ObjectRelationship).where(ObjectRelationship.frame_id == frame_id,
                                         ObjectRelationship.kind == "occludes"))).scalars().all()}

    created = 0
    for p in plan["pairs"]:
        key = (p["from_object_id"], p["to_object_id"])
        if key in existing:
            continue
        db.add(ObjectRelationship(
            from_object_id=UUID(p["from_object_id"]), to_object_id=UUID(p["to_object_id"]),
            frame_id=frame_id, kind="occludes", status="proposed", source="geometry",
            conf=p["conf"],
            # What the claim rests on, so a reviewer can weigh it rather than take it on trust.
            evidence={"overlap": p["overlap"], "depth_gap_m": p["depth_gap_m"],
                      "near_m": p["near_m"], "far_m": p["far_m"], "method": "ipm_mono_v1"}))
        created += 1
    log.info("occlusion.committed", frame=str(frame_id)[:8], created=created,
             unordered=len(plan["unordered"]))
    return {**plan, "created": created}


async def depth_order(db: AsyncSession, frame_id: UUID) -> dict:
    """The frame's objects from nearest to furthest, for a renderer that wants to draw back to front.

    A total order over the objects that HAVE a depth, and a separate list of those that do not. Merging
    the two would put every unmeasured object at one end of the order, which is a statement about them
    that nothing measured.
    """
    rows = (await db.execute(
        select(Object.object_id, ObjectDynamics.distance_m)
        .join(ObjectDynamics, ObjectDynamics.object_id == Object.object_id, isouter=True)
        .where(Object.frame_id == frame_id, Object.state != "rejected"))).all()
    with_depth = [(str(r.object_id), float(r.distance_m)) for r in rows if r.distance_m is not None]
    with_depth.sort(key=lambda t: t[1])
    return {
        "frame_id": str(frame_id),
        "order": [oid for oid, _ in with_depth],
        "depth_m": {oid: round(d, 2) for oid, d in with_depth},
        "no_depth": [str(r.object_id) for r in rows if r.distance_m is None],
    }
