"""Which object on this frame is most likely to be wrong, and how much of that is actually known.

Every ingredient for this already existed and none of them were ever put together. `triage.py` ranks by
uncertainty times rarity but has no frame filter. `selector.py` composes seven signals and stops at
session scope. `quality_score.py` scores one object at a time. `CliqueBandit` allocates budget across
confusion cliques and never touches an individual object. `ego_mask.py` answers a yes/no and deletes.
Meanwhile `GET /api/frames/{id}/objects` returns objects in whatever order Postgres produced them.

Two decisions shape this module, and both are about honesty rather than ranking.

**Order is a separate field, never a reordering.** `web/components/editor/properties/objectGroups.ts`
records that the server's order is drawing order and that re-sorting the rows silently breaks the
correspondence between the list and the canvas. So `rank_frame_objects` returns a per-object score and the
caller decides what to do with it; the array it was given comes back in the order it arrived.

**A missing signal is reported, not assumed.** Coverage over the corpus is uneven and in one case zero:

    calibrated confidence   89.5%
    entropy                 86.0%
    proposals               89.5%
    mask/box disagreement   13.5% carry the flag
    quality_score           11.9%
    clique mass             0 rows in clique_bandit
    ego mask                only where one has been estimated for that camera

A score that treats an absent component as zero ranks an unmeasured object as safe, which is the worst
direction to be wrong in for a queue whose whole purpose is to surface what nobody has checked. So each
component reports whether it fired, the risk is the weighted mean over the ones that did, and every row
carries the list of components that contributed plus the fraction of the total weight they represent.
An object scored on two of eight signals says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object, ObjectRelationship
from db.models import Session as DbSession

log = get_logger("frame_risk")

# Relative weight of each component. Ordered by how directly it has been shown to predict a wrong label
# here: a calibrated confidence and a path disagreement are measured signals, rarity is a prior, and the
# ego-mask term is a specific and narrow failure shape.
WEIGHTS = {
    "calibrated_conf": 1.0,
    "entropy": 0.8,
    "path_disagreement": 0.8,
    "mask_box": 0.7,
    "quality": 0.7,
    "rarity": 0.5,
    "ego_edge": 0.5,
    "provisional_relations": 0.3,
}

# Below this a detection is not trusted on its own anywhere else in the repo, so it is where the
# confidence term starts contributing rather than at 1.0.
LOW_CONF_FLOOR = 0.6


@dataclass
class Component:
    """One signal's contribution, and whether it was actually available for this object."""
    name: str
    value: float          # 0.0 (unremarkable) to 1.0 (as risky as this signal gets)
    present: bool
    detail: str = ""


@dataclass
class ObjectRisk:
    object_id: str
    risk: float
    # Which signals contributed, and how much of the total weight they carry. A row ranked on a quarter of
    # the weight is a different claim from one ranked on all of it, and the client shows the difference.
    components: list[Component] = field(default_factory=list)
    coverage: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "risk": round(self.risk, 4),
            "coverage": round(self.coverage, 3),
            "reasons": self.reasons,
            "components": {c.name: {"value": round(c.value, 4), "present": c.present, "detail": c.detail}
                           for c in self.components},
        }


def _calibrated(prov: dict, conf: float | None) -> Component:
    """Low confidence, on the calibrated number where one exists.

    `calibrated_from` holds the raw score the calibration mapped, so its presence is what says the object
    went through calibration at all; `conf` is then the calibrated value.
    """
    if conf is None:
        return Component("calibrated_conf", 0.0, False, "no confidence on the object")
    calibrated = "calibrated_from" in prov
    # Scaled so the floor is where risk starts, rather than spreading the whole 0..1 range and calling a
    # 0.9-confidence detection mildly risky.
    value = max(0.0, min(1.0, (LOW_CONF_FLOOR - float(conf)) / LOW_CONF_FLOOR))
    return Component("calibrated_conf", value, True,
                     f"conf {float(conf):.2f}{'' if calibrated else ' (uncalibrated)'}")


def _entropy(prov: dict) -> Component:
    e = prov.get("entropy")
    if not isinstance(e, int | float) or isinstance(e, bool):
        return Component("entropy", 0.0, False, "not scored")
    # Entropy over the path distribution. Normalised against 1.0, which is where the observed values sit;
    # anything above is clamped rather than allowed to dominate every other term.
    return Component("entropy", max(0.0, min(1.0, float(e))), True, f"entropy {float(e):.2f}")


def _paths(prov: dict) -> Component:
    props = prov.get("proposals")
    if not isinstance(props, list) or len(props) < 2:
        return Component("path_disagreement", 0.0, False,
                         "fewer than two paths proposed this object")
    overruled = sum(1 for p in props if isinstance(p, dict) and p.get("verdict") == "overruled")
    names = {p.get("class_name") for p in props if isinstance(p, dict) and p.get("class_name")}
    value = 1.0 if len(names) > 1 else (0.6 if overruled else 0.0)
    detail = (f"{len(names)} classes proposed" if len(names) > 1
              else f"{overruled} overruled" if overruled else "paths agreed")
    return Component("path_disagreement", value, True, detail)


def _mask_box(prov: dict) -> Component:
    v = prov.get("mask_box_disagree")
    if v is None:
        return Component("mask_box", 0.0, False, "not measured")
    return Component("mask_box", 1.0 if v else 0.0, True,
                     "outline does not match the box" if v else "outline matches")


def _quality(score: float | None) -> Component:
    if score is None:
        return Component("quality", 0.0, False, "never scored")
    return Component("quality", max(0.0, min(1.0, 1.0 - float(score))), True, f"quality {float(score):.2f}")


def _rarity(onto: Any, class_id: int) -> Component:
    try:
        c = onto.by_id(class_id)
    except KeyError:
        # A class the ontology does not know is itself worth looking at, and saying "not present" would
        # rank it as unremarkable.
        return Component("rarity", 1.0, True, f"class {class_id} is not in the ontology")
    rare = bool(c.india or c.l1 == "fallback")
    return Component("rarity", 1.0 if rare else 0.0, True,
                     "India-specific or fallback class" if rare else c.name)


def _ego_edge(mask, bbox, w: float, h: float) -> Component:
    """A box half on the bonnet, which is where a reflection or a piece of the car gets read as an object.

    Peaked in the middle rather than monotone: fully on the hood is not a ranking problem, because
    `cleanup_sweep` already deletes those, and fully off it is unremarkable.
    """
    if mask is None or not w or not h:
        return Component("ego_edge", 0.0, False, "no ego mask estimated for this camera")
    frac = mask.ego_fraction(tuple(float(v) for v in bbox), float(w), float(h))
    value = max(0.0, 1.0 - abs(frac - 0.5) * 2.0)
    return Component("ego_edge", value, True, f"{frac * 100:.0f}% of the box is on the ego hood")


def _relations(n_proposed: int) -> Component:
    if n_proposed <= 0:
        # Genuinely zero rather than unmeasured: the query ran and found none. Distinguishing them would
        # be dishonest in the other direction, since `object_relationship` holds 98 rows corpus-wide and
        # treating every object as "measured, no relations" would let a signal nobody collects carry
        # weight it has not earned.
        return Component("provisional_relations", 0.0, False, "no relations proposed on this object")
    return Component("provisional_relations", min(1.0, n_proposed / 3.0), True,
                     f"{n_proposed} unconfirmed relation(s)")


def score_object(obj: Object, onto: Any, *, ego_mask=None, frame_w: float | None = None,
                 frame_h: float | None = None, n_proposed_relations: int = 0) -> ObjectRisk:
    """Compose the available signals into one risk for one object."""
    prov = obj.provenance or {}
    comps = [
        _calibrated(prov, obj.conf),
        _entropy(prov),
        _paths(prov),
        _mask_box(prov),
        _quality(obj.quality_score),
        _rarity(onto, obj.class_id),
        _ego_edge(ego_mask, obj.bbox, frame_w or 0.0, frame_h or 0.0),
        _relations(n_proposed_relations),
    ]
    live = [c for c in comps if c.present]
    total_w = sum(WEIGHTS[c.name] for c in live)
    risk = (sum(WEIGHTS[c.name] * c.value for c in live) / total_w) if total_w else 0.0
    coverage = total_w / sum(WEIGHTS.values())
    # Only the signals that actually fired, so the reason list explains the rank rather than listing
    # everything that was looked at.
    reasons = [c.detail for c in sorted(live, key=lambda c: -WEIGHTS[c.name] * c.value)
               if c.value > 0 and c.detail]
    return ObjectRisk(str(obj.object_id), risk, comps, coverage, reasons)


async def rank_frame_objects(db: AsyncSession, frame_id: UUID, onto: Any) -> dict:
    """Risk for every object on one frame, plus what the ranking was able to see.

    Returns scores keyed by object id and a `ranking` list ordered by risk. It does NOT return the objects
    themselves and it does not reorder anything: the editor's object list is in drawing order and stays
    that way.
    """
    frame = await db.get(Frame, frame_id)
    if frame is None:
        return {"frame_id": str(frame_id), "objects": {}, "ranking": [], "reason": "frame not found"}

    objs = list((await db.execute(select(Object).where(Object.frame_id == frame_id))).scalars().all())
    if not objs:
        return {"frame_id": str(frame_id), "objects": {}, "ranking": [], "n": 0}

    # Proposed relations per object, in one query rather than one per object.
    rel_counts: dict[Any, int] = {}
    rows = (await db.execute(
        select(ObjectRelationship.from_object_id, func.count())
        .where(ObjectRelationship.frame_id == frame_id, ObjectRelationship.status == "proposed")
        .group_by(ObjectRelationship.from_object_id))).all()
    for oid, n in rows:
        rel_counts[oid] = int(n)

    # The ego mask is per camera and cached in the object store, so it is fetched once for the frame.
    ego = None
    if frame.cam_id:
        sess = await db.get(DbSession, frame.session_id)
        if sess is not None and sess.vehicle_id:
            try:
                from services.autolabel.ego_mask import get_ego_mask

                ego = get_ego_mask(sess.vehicle_id, frame.cam_id)
            except Exception as exc:  # noqa: BLE001
                # A missing or unreadable mask is one absent component, not a failed ranking. Logged so
                # that "no ego mask on any frame" is findable rather than looking like a design choice.
                log.info("frame_risk.no_ego_mask", frame=str(frame_id)[:8], reason=str(exc)[:120])

    scored = [score_object(o, onto, ego_mask=ego, frame_w=frame.width, frame_h=frame.height,
                           n_proposed_relations=rel_counts.get(o.object_id, 0)) for o in objs]
    ranked = sorted(scored, key=lambda r: (-r.risk, r.object_id))
    return {
        "frame_id": str(frame_id),
        "n": len(scored),
        "objects": {r.object_id: r.as_dict() for r in scored},
        # Ranked ids only. The caller already has the objects and re-sending them here would invite a
        # client to render THIS order, which is the thing objectGroups.ts warns against.
        "ranking": [r.object_id for r in ranked],
        # How much of the weight the ranking could actually see, averaged over the frame. A frame that
        # reads 0.3 is not a confident ordering and the editor says so rather than implying one.
        "coverage": round(sum(r.coverage for r in scored) / len(scored), 3),
        "components_available": sorted({c.name for r in scored for c in r.components if c.present}),
        "components_missing": sorted({c.name for r in scored for c in r.components} -
                                     {c.name for r in scored for c in r.components if c.present}),
    }
