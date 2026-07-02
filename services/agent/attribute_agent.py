"""Auto-attribute fill: the mechanical attributes that are derivable from geometry and motion get filled
automatically, so annotators only touch the ones that need eyes (colour, helmet, livery).

Derived without any model:
- occlusion (0/25/50/75/100): the fraction of this box covered by nearer objects (nearer = lower in the
  image, the depth proxy for a forward camera), quantized to the ontology's steps.
- truncation (0..1): how much the box is cut off by a frame edge (0 when it sits clear of the border).
- static (bool) and direction (same/cross/wrong_way): read from the object's derived dynamics (speed and
  heading) when a dynamics pass has run.

Only attributes the object's class actually allows are filled, only when currently empty, and every value is
validated against the ontology before it is written. One reversible AgentRun records the prior attrs so
revert restores them exactly. Appearance attributes are intentionally left for a human (or the VLM path).
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Frame, Object, ObjectDynamics

log = get_logger("agent.attribute")

_OCC_STEPS = [0, 25, 50, 75, 100]


def _quantize_occ(frac: float) -> int:
    pct = max(0.0, min(1.0, frac)) * 100.0
    return min(_OCC_STEPS, key=lambda s: abs(s - pct))


def _occlusion(box, others) -> int:
    """Fraction of `box` covered by nearer boxes (bottom lower in the image), quantized."""
    x1, y1, x2, y2 = (float(v) for v in box)
    area = max(1.0, (x2 - x1) * (y2 - y1))
    covered = 0.0
    for ob in others:
        bx1, by1, bx2, by2 = (float(v) for v in ob.bbox)
        if by2 <= y2:  # not nearer than this object
            continue
        ix1, iy1 = max(x1, bx1), max(y1, by1)
        ix2, iy2 = min(x2, bx2), min(y2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        covered += iw * ih
    return _quantize_occ(min(1.0, covered / area))


def _truncation(box, w: int, h: int) -> float:
    """Coarse fraction of the box lost to a frame edge (0 when clear of the border)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    m = 1.5
    edges = sum([x1 <= m, y1 <= m, x2 >= w - m, y2 >= h - m])
    return round(min(0.6, edges * 0.2), 2)


def _from_dynamics(dyn) -> dict:
    out: dict = {}
    spd = getattr(dyn, "speed_kmh", None)
    if spd is not None:
        out["static"] = bool(spd < 2.0)
    hd = getattr(dyn, "heading_deg", None)
    if hd is not None:
        a = abs(((float(hd) + 180.0) % 360.0) - 180.0)  # fold to [0,180]
        out["direction"] = "same" if a < 45 else "wrong_way" if a > 135 else "cross"
    return out


async def plan_attributes(db: AsyncSession, frame_id: uuid.UUID) -> dict:
    """Dry-run: the derivable attributes each machine object would gain. No writes."""
    from services.autolabel.ontology import get_ontology

    frame = await db.get(Frame, frame_id)
    if frame is None:
        raise ValueError("frame not found")
    onto = get_ontology()
    objs = (await db.execute(select(Object).where(Object.frame_id == frame_id, Object.source != "human"))).scalars().all()
    dyn = {str(d.object_id): d for d in (await db.execute(
        select(ObjectDynamics).where(ObjectDynamics.object_id.in_([o.object_id for o in objs])))).scalars().all()} if objs else {}

    items = []
    counts = {"objects": 0, "attrs_filled": 0, "by_attr": {}}
    for o in objs:
        allowed = onto.attrs_for_class(o.class_id)
        cur = o.attrs or {}
        proposed: dict = {}
        if (allowed is None or "occlusion" in allowed) and "occlusion" not in cur:
            proposed["occlusion"] = _occlusion(o.bbox, [x for x in objs if x.object_id != o.object_id])
        if (allowed is None or "truncation" in allowed) and "truncation" not in cur:
            proposed["truncation"] = _truncation(o.bbox, frame.width, frame.height)
        d = dyn.get(str(o.object_id))
        if d is not None:
            for k, v in _from_dynamics(d).items():
                if (allowed is None or k in allowed) and k not in cur:
                    proposed[k] = v
        # keep only values that validate for this class
        proposed = {k: v for k, v in proposed.items() if not onto.validate_attrs({k: v}, o.class_id)}
        if not proposed:
            continue
        counts["objects"] += 1
        for k in proposed:
            counts["attrs_filled"] += 1
            counts["by_attr"][k] = counts["by_attr"].get(k, 0) + 1
        try:
            cname = onto.by_id(int(o.class_id)).name
        except Exception:  # noqa: BLE001
            cname = str(o.class_id)
        items.append({"object_id": str(o.object_id), "class_name": cname, "attrs": proposed})
    return {"frame_id": str(frame_id), "counts": counts, "items": items}


async def commit_attributes(db: AsyncSession, frame_id: uuid.UUID, created_by: str | None = None) -> dict:
    """Fill the derivable attributes on the objects as one reversible run (revert restores prior attrs)."""
    plan = await plan_attributes(db, frame_id)
    run_id = uuid.uuid4()
    changes: dict[str, dict] = {}
    for item in plan["items"]:
        obj = await db.get(Object, uuid.UUID(item["object_id"]))
        if obj is None or obj.source == "human":
            continue
        prior = dict(obj.attrs or {})
        changes[item["object_id"]] = {"from_attrs": prior}
        obj.attrs = {**prior, **item["attrs"]}
        obj.version = (obj.version or 0) + 1
        prov = dict(obj.provenance or {})
        prov["agent_run_id"] = str(run_id)
        prov.setdefault("agent_attrs", []).extend(item["attrs"].keys())
        obj.provenance = prov
    db.add(AgentRun(run_id=run_id, kind="attribute", scope={"frame_id": str(frame_id)}, status="committed",
                    policy={}, counts=plan["counts"], changes=changes, critic={}, created_by=created_by))
    await db.commit()
    log.info("agent.attribute.commit", frame_id=str(frame_id), run_id=str(run_id), objects=len(changes))
    return {"run_id": str(run_id), "frame_id": str(frame_id), "objects_updated": len(changes), "counts": plan["counts"]}
