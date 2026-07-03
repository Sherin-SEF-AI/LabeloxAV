"""Scene-graph relations (M-F.5, part 1): typed, closed relations between the objects on a frame, the edges an
AV-2.0 dataset needs on top of per-object labels. They are auto-proposed from geometry where computable
(occluded_by, following, parked_near, crossing_in_front_of) and left for the VLM or a human otherwise
(yielding_to, stopping_at), and every proposal is confirmed by a human before it counts. Stored on the existing
ObjectRelationship with status/source/evidence.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object, ObjectRelationship
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology

log = get_logger("intelligence.scene_graph")

# closed scene-graph relation vocabulary
SCENE_RELATIONS = ["following", "yielding_to", "occluded_by", "stopping_at", "parked_near", "crossing_in_front_of"]
_GEOMETRIC = {"occluded_by", "following", "parked_near", "crossing_in_front_of"}
_VEHICLE_L1 = {"two_wheeler", "three_wheeler", "four_wheeler", "heavy"}
_VRU_L1 = {"vru"}


def vocab() -> dict:
    return {"relations": SCENE_RELATIONS, "geometric": sorted(_GEOMETRIC),
            "vlm_or_human": sorted(set(SCENE_RELATIONS) - _GEOMETRIC)}


def _iom(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    m = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return inter / m if m > 0 else 0.0


def propose_from_geometry(objs: list[dict], w: float, h: float, cap: int = 40) -> list[dict]:
    """Geometric scene-graph proposals: for each object, its SINGLE best partner per relation kind (not every
    qualifying pair, which explodes on dense frames). Directed: from -> to. Capped per frame."""
    g = []
    for o in objs:
        bb = o["bbox"]
        g.append({"id": o["id"], "bbox": bb, "cx": (bb[0] + bb[2]) / 2 / w, "by": bb[3] / h,
                  "area": (bb[2] - bb[0]) * (bb[3] - bb[1]), "veh": o["l1"] in _VEHICLE_L1,
                  "vru": o["l1"] in _VRU_L1})
    out: list[dict] = []
    parked_seen: set = set()
    for i, a in enumerate(g):
        best: dict = {}  # kind -> (score, record) keeping the strongest single partner
        for j, b in enumerate(g):
            if i == j:
                continue
            overlap = _iom(a["bbox"], b["bbox"])
            col = abs(a["cx"] - b["cx"])
            # occluded_by: a behind b (b nearer/lower, larger, strongly overlapping) -> keep the max overlap
            if overlap > 0.45 and b["by"] > a["by"] + 0.03 and b["area"] > a["area"] * 1.1:
                if overlap > best.get("occluded_by", (0,))[0]:
                    best["occluded_by"] = (overlap, {"from": a["id"], "to": b["id"], "kind": "occluded_by",
                                                     "evidence": {"overlap": round(overlap, 2)}})
            # following: nearest vehicle directly ahead in a's lane -> keep the smallest gap
            if a["veh"] and b["veh"] and col < 0.05 and a["by"] > b["by"] + 0.06 and overlap < 0.3:
                gap = a["by"] - b["by"]
                if (1.0 - gap) > best.get("following", (0,))[0]:
                    best["following"] = (1.0 - gap, {"from": a["id"], "to": b["id"], "kind": "following",
                                                     "evidence": {"lane_align": round(col, 3)}})
            # crossing_in_front_of: nearest vehicle behind a VRU that is ahead of it in the same column
            if a["vru"] and b["veh"] and a["by"] < b["by"] - 0.04 and col < 0.10:
                score = 1.0 - col
                if score > best.get("crossing_in_front_of", (0,))[0]:
                    best["crossing_in_front_of"] = (score, {"from": a["id"], "to": b["id"],
                                                            "kind": "crossing_in_front_of",
                                                            "evidence": {"ahead_by": round(b["by"] - a["by"], 3)}})
            # parked_near: nearest vehicle beside a (tight, non-overlapping) -> keep the closest
            if a["veh"] and b["veh"] and overlap < 0.08 and col < 0.10 and abs(a["by"] - b["by"]) < 0.06:
                score = 1.0 - col
                if score > best.get("parked_near", (0,))[0]:
                    best["parked_near"] = (score, {"from": a["id"], "to": b["id"], "kind": "parked_near",
                                                   "evidence": {"centre_gap": round(col, 3)}})
        for kind, (_score, rec) in best.items():
            if kind == "parked_near":
                key = tuple(sorted((str(rec["from"]), str(rec["to"]))))
                if key in parked_seen:
                    continue
                parked_seen.add(key)
            out.append(rec)
    out.sort(key=lambda r: r["kind"])
    return out[:cap]


async def propose_relations(frame_id: UUID, db: AsyncSession | None = None) -> dict:
    """Propose geometric scene-graph relations for a frame (idempotent: skips relations already present)."""
    own = db is None
    maker = get_sessionmaker()
    db = db or maker()
    try:
        onto = get_ontology()
        fr = await db.get(Frame, frame_id)
        if fr is None:
            return {"error": "frame not found"}
        rows = (await db.execute(select(Object).where(Object.frame_id == frame_id, Object.state != "rejected"))).scalars().all()
        objs = [{"id": o.object_id, "bbox": [float(x) for x in o.bbox],
                 "l1": _l1(o.class_id, onto)} for o in rows]
        proposals = propose_from_geometry(objs, float(fr.width or 1920), float(fr.height or 1080))

        existing = {(r.from_object_id, r.to_object_id, r.kind) for r in
                    (await db.execute(select(ObjectRelationship).where(ObjectRelationship.frame_id == frame_id))).scalars().all()}
        added = 0
        for p in proposals:
            key = (p["from"], p["to"], p["kind"])
            if key in existing:
                continue
            db.add(ObjectRelationship(from_object_id=p["from"], to_object_id=p["to"], frame_id=frame_id,
                                      kind=p["kind"], status="proposed", source="geometry", evidence=p["evidence"]))
            added += 1
        await db.commit()
        return {"frame_id": str(frame_id), "proposed": added,
                "by_kind": _counts([p["kind"] for p in proposals])}
    finally:
        if own:
            await db.close()


def _l1(class_id: int, onto) -> str:
    try:
        return onto.by_id(class_id).l1
    except Exception:  # noqa: BLE001
        return ""


def _counts(items: list[str]) -> dict:
    out: dict = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


async def set_relation_status(relationship_id: UUID, status: str) -> dict:
    """Confirm or reject a proposed scene-graph relation."""
    if status not in ("confirmed", "rejected", "proposed"):
        return {"error": "status must be confirmed | rejected | proposed"}
    maker = get_sessionmaker()
    async with maker() as db:
        rel = await db.get(ObjectRelationship, relationship_id)
        if rel is None:
            return {"error": "relationship not found"}
        if status == "rejected":
            await db.delete(rel)
            await db.commit()
            return {"relationship_id": str(relationship_id), "status": "rejected"}
        rel.status = status
        await db.commit()
        return {"relationship_id": str(relationship_id), "status": status}


async def frame_relations(frame_id: UUID) -> list[dict]:
    """Every relation on a frame with its status and source, for the scene-graph UI."""
    maker = get_sessionmaker()
    onto = get_ontology()
    async with maker() as db:
        rels = (await db.execute(select(ObjectRelationship).where(ObjectRelationship.frame_id == frame_id))).scalars().all()
        objs = {o.object_id: o.class_id for o in
                (await db.execute(select(Object).where(Object.frame_id == frame_id))).scalars().all()}

        def name(oid):
            cid = objs.get(oid)
            try:
                return onto.by_id(cid).name if cid is not None else "?"
            except Exception:  # noqa: BLE001
                return "?"

        return [{"relationship_id": str(r.relationship_id), "from_object_id": str(r.from_object_id),
                 "to_object_id": str(r.to_object_id), "from_name": name(r.from_object_id),
                 "to_name": name(r.to_object_id), "kind": r.kind, "status": r.status, "source": r.source,
                 "evidence": r.evidence or {}} for r in rels]
