"""Why a check thinks a label is wrong, in a form that can be drawn on the frame (tool #12).

Every detector already leaves its numbers in `ErrorCandidate.detail`, and the queue showed a score and a
detector name. A reviewer then had to know what "confident_learning 0.81" means and go and find the frame
to judge it. This turns a candidate into an explanation a person can check against the pixels:

  reason      one plain sentence: what the check saw and why that makes the label suspect
  evidence    boxes on the object's own frame: the object itself, plus whatever the check compared it to
              there (the box it duplicates, the vehicle a rider is inside)
  neighbours  the objects elsewhere that the check leaned on, with crops: the embedding neighbours a
              classifier learned from, the rest of a track, the same object in another camera
  twin frame  for the near-duplicate check, the look-alike frame and what is labelled on it
  facts       the numbers behind the sentence, labelled

Built on read from the candidate rather than stored, so the 300k candidates already in the queue are
explained without a re-run, and an explanation never goes stale against an object that has since moved.
Where a detector already recorded who its neighbours were (confident learning stores them now) those are
used as recorded; otherwise they are found the way the detector would have found them.

Nothing here is a new judgement. Every sentence restates what a detector measured; none of them claims the
label is wrong, because none of the detectors is calibrated until people rule on it
(`queue.detector_precision`).
"""

from __future__ import annotations

import uuid
from collections import Counter

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import ErrorCandidate, Frame, Object, ObjectEmbedding

log = get_logger("errordetect.explain")

# The same words /quality/errors uses, so the page and the canvas describe a check identically.
DETECTOR_NAMES = {
    "confident_learning": "Disagrees with the model",
    "embedding_outlier": "Looks unlike its class",
    "track_inconsistent": "Class changes along its track",
    "track_consistency": "Class changes along its track",
    "cross_cam_inconsistent": "Disagrees across cameras",
    "critic_flag": "Flagged by the AI critic",
    "near_dup_inconsistent": "Differs from near-duplicates",
    "policy_violation": "Breaks a labelling rule",
    "reanalyze": "Flagged on re-check",
}

# Rules whose evidence is another box on the same frame, and which box it is.
_OVERLAP_RULES = {"duplicate", "duplicate_box"}
_CONTAINER_RULES = {"vru_inside_vehicle", "part_of_vehicle"}

_RULE_TEXT = {
    "min_box_size": "The box is smaller than the guideline's minimum, which is almost always noise.",
    "degenerate_aspect": "The box's shape is implausible for its class.",
    "attr_validity": "Its attributes are invalid or do not apply to its class.",
    "duplicate_box": "Another box of the same class covers nearly the same pixels.",
    "duplicate": "Another box on this frame covers the same object with higher confidence.",
    "impossible_size": "Its size is impossible for its class at this position in the image.",
    "smaller_than_vru": "It is smaller than any pedestrian or rider could be here.",
    "above_horizon": "A ground object sits above the horizon.",
    "vru_inside_vehicle": "A pedestrian or rider box sits inside a vehicle box.",
    "part_of_vehicle": "The box covers part of a vehicle rather than a whole object.",
    "stuff": "The box covers background (road, sky, building) rather than a countable object.",
    "critic_flag": "The consistency critic flagged it.",
}


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _inside_frac(inner: list[float], outer: list[float]) -> float:
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def _box(b) -> list[float]:
    return [round(float(v), 1) for v in list(b or [])[:4]]


def _pct(x) -> str:
    try:
        return f"{float(x) * 100:.0f}%"
    except (TypeError, ValueError):
        return "?"


class _Names:
    def __init__(self) -> None:
        from services.autolabel.ontology import get_ontology

        self.onto = get_ontology()

    def __call__(self, class_id) -> str:
        try:
            return self.onto.by_id(int(class_id)).name
        except Exception:  # noqa: BLE001
            return str(class_id)


def _neighbour(o: Object, names: _Names, role: str, similarity: float | None = None) -> dict:
    return {"object_id": str(o.object_id), "frame_id": str(o.frame_id), "class_name": names(o.class_id),
            "bbox": _box(o.bbox), "similarity": round(similarity, 4) if similarity is not None else None,
            "role": role, "crop_url": f"/api/objects/{o.object_id}/crop"}


async def _embedding_neighbours(db: AsyncSession, obj: Object, names: _Names, k: int = 6,
                                stored: list | None = None) -> list[dict]:
    """The nearest crops in DINOv3 space: what an embedding-based check compares this object against."""
    pairs: list[tuple[str, float | None]] = []
    if stored:
        pairs = [(str(n["object_id"]), n.get("sim")) for n in stored if n.get("object_id")][:k]
    else:
        emb = await db.get(ObjectEmbedding, obj.object_id)
        if emb is None or emb.dino_vec is None:
            return []
        from core.embeddings import object_neighbors

        pairs = await object_neighbors(db, emb.dino_vec, k=k, exclude_object_id=obj.object_id)
    if not pairs:
        return []
    rows = {str(o.object_id): o for o in (await db.execute(
        select(Object).where(Object.object_id.in_([uuid.UUID(p[0]) for p in pairs])))).scalars().all()}
    return [_neighbour(rows[oid], names, "embedding neighbour", sim) for oid, sim in pairs if oid in rows]


def _h(name) -> str:
    return str(name).replace("_", " ")


def _class_tally(neighbours: list[dict]) -> str:
    c = Counter(n["class_name"] for n in neighbours)
    return ", ".join(f"{_h(name)} x{n}" for name, n in c.most_common(4))


async def explain(db: AsyncSession, cand: ErrorCandidate, *, obj: Object | None = None,
                  frame: Frame | None = None, frame_objects: list[Object] | None = None,
                  names: _Names | None = None) -> dict | None:
    """One candidate, explained. None when its object is gone."""
    names = names or _Names()
    obj = obj or await db.get(Object, cand.object_id)
    if obj is None:
        return None
    frame = frame or await db.get(Frame, obj.frame_id)
    d = dict(cand.detail or {})
    cls = names(obj.class_id)
    proposed = (cand.proposed_label or {}).get("class_name")
    reason = ""
    facts: list[dict] = []
    evidence: list[dict] = [{"bbox": _box(obj.bbox), "role": "subject", "label": cls,
                             "object_id": str(obj.object_id)}]
    neighbours: list[dict] = []
    twin: dict | None = None
    kind = cand.kind
    rule = d.get("rule")

    async def same_frame() -> list[Object]:
        if frame_objects is not None:
            return frame_objects
        return list((await db.execute(select(Object).where(Object.frame_id == obj.frame_id))).scalars().all())

    if kind == "confident_learning":
        given_p, prop_p = d.get("given_prob"), d.get("proposed_prob")
        neighbours = await _embedding_neighbours(db, obj, names, stored=d.get("neighbours"))
        reason = (f"A classifier trained on the other labelled crops thinks this looks like "
                  f"{_h(proposed) if proposed else 'another class'} ({_pct(prop_p)}) and gives {_h(cls)} only "
                  f"{_pct(given_p)}.")
        if neighbours:
            reason += f" Its closest crops are labelled {_class_tally(neighbours)}."
        facts = [{"label": f"P({cls})", "value": _pct(given_p)},
                 {"label": f"P({proposed})" if proposed else "P(proposed)", "value": _pct(prop_p)}]
    elif kind == "embedding_outlier":
        dist, thr = d.get("centroid_distance"), d.get("class_threshold")
        neighbours = await _embedding_neighbours(db, obj, names)
        reason = (f"It looks less like a typical {_h(cls)} than 97% of {_h(cls)} labels do "
                  f"(distance {dist} from the class centre; the cut-off is {thr}).")
        if neighbours:
            reason += f" The crops it most resembles are labelled {_class_tally(neighbours)}."
        facts = [{"label": "distance from class centre", "value": str(dist)},
                 {"label": "class cut-off", "value": str(thr)}]
    elif kind in ("track_inconsistent", "track_consistency"):
        maj, n = d.get("track_majority"), d.get("track_len")
        tid = d.get("track_id") or (str(obj.track_id) if obj.track_id else None)
        if tid:
            mates = list((await db.execute(
                select(Object, Frame.ts_ns).join(Frame, Frame.frame_id == Object.frame_id)
                .where(Object.track_id == uuid.UUID(str(tid)), Object.object_id != obj.object_id))).all())
            t0 = frame.ts_ns if frame is not None else 0
            mates.sort(key=lambda r: abs((r[1] or 0) - t0))
            neighbours = [_neighbour(o, names, "same track") for o, _ in mates[:6]]
        given = d.get("given_class") or cls
        reason = (f"Most of its track ({n} boxes) is labelled {_h(maj)}; this box said {_h(given)}. A class that "
                  f"changes for one frame along a track is usually a slip.")
        if given != cls:
            reason += f" It has since been changed to {_h(cls)}."
        facts = [{"label": "track length", "value": str(n)}, {"label": "track majority", "value": str(maj)}]
    elif kind == "cross_cam_inconsistent":
        maj, nv = d.get("rig_majority"), d.get("n_views")
        rid = d.get("rig_track_id") or (str(obj.rig_track_id) if obj.rig_track_id else None)
        if rid:
            others = list((await db.execute(
                select(Object).where(Object.rig_track_id == uuid.UUID(str(rid)),
                                     Object.object_id != obj.object_id).limit(6))).scalars().all())
            neighbours = [_neighbour(o, names, "other camera") for o in others]
        reason = (f"The same object seen by {nv} cameras is mostly labelled {_h(maj)}; in this view it is {_h(cls)}.")
        facts = [{"label": "views", "value": str(nv)}, {"label": "majority in other views", "value": str(maj)}]
    elif kind == "near_dup_inconsistent":
        sim = d.get("similarity")
        twin_id = d.get("near_dup_frame")
        reason = (f"A frame {_pct(sim)} identical to this one has no {_h(cls)} on it. A detection that appears "
                  f"on one of two near-identical frames and not the other is often a false positive.")
        if twin_id:
            tobjs = list((await db.execute(
                select(Object).where(Object.frame_id == uuid.UUID(str(twin_id))))).scalars().all())
            tframe = await db.get(Frame, uuid.UUID(str(twin_id)))
            twin = {"frame_id": str(twin_id), "similarity": sim,
                    "width": tframe.width if tframe else None, "height": tframe.height if tframe else None,
                    "image_url": f"/api/frames/{twin_id}/image",
                    "objects": [{"bbox": _box(o.bbox), "class_name": names(o.class_id),
                                 "object_id": str(o.object_id)} for o in tobjs[:60]]}
        facts = [{"label": "frame similarity", "value": _pct(sim)},
                 {"label": "object confidence", "value": str(d.get("conf"))}]
    elif kind == "critic_flag":
        checks = d.get("checks") or []
        reasons = d.get("reasons") or []
        if isinstance(reasons, dict):
            reasons = list(reasons.values())
        reason = "; ".join(str(r) for r in reasons[:3]) or "The consistency critic flagged it."
        facts = [{"label": "checks", "value": ", ".join(map(str, checks))}]
    else:  # policy_violation, reanalyze and anything newer that follows their {rule, reason} shape
        stated = str(d.get("reason") or "")
        # The re-check's generic "quality review flagged 'x'" restates the rule name; the rule's own
        # sentence says what it means.
        generic = not stated or stated.startswith("quality review flagged")
        rule_text = _RULE_TEXT.get(rule or "")
        if generic:
            reason = rule_text or stated
        elif rule_text and "rule" in stated:
            # A terse machine phrase ("cleanup rule 'stuff' applies") names the rule; say what it means too.
            reason = f"{rule_text} ({stated})"
        else:
            reason = stated
        reason = reason or "A labelling rule flagged it."
        b = [float(v) for v in (obj.bbox or [0, 0, 0, 0])[:4]]
        if rule in _OVERLAP_RULES or rule in _CONTAINER_RULES:
            best, best_v = None, 0.0
            for o in await same_frame():
                if o.object_id == obj.object_id or not o.bbox or len(o.bbox) < 4:
                    continue
                ob = [float(v) for v in o.bbox[:4]]
                if rule in _OVERLAP_RULES:
                    if rule == "duplicate_box" and int(o.class_id) != int(obj.class_id):
                        continue
                    v = _iou(b, ob)
                else:
                    v = _inside_frac(b, ob) if (ob[2] - ob[0]) * (ob[3] - ob[1]) > (b[2] - b[0]) * (b[3] - b[1]) else 0.0
                if v > best_v:
                    best, best_v = o, v
            if best is not None:
                role = "overlaps" if rule in _OVERLAP_RULES else "contains"
                evidence.append({"bbox": _box(best.bbox), "role": role,
                                 "label": f"{names(best.class_id)} ({'IoU' if role == 'overlaps' else 'covers'} {best_v:.2f})",
                                 "object_id": str(best.object_id)})
                neighbours = [_neighbour(best, names, role, None)]
                facts.append({"label": "IoU with the other box" if role == "overlaps" else "share inside the other box",
                              "value": f"{best_v:.2f}"})
        if frame is not None:
            facts.append({"label": "box size", "value": f"{int(b[2] - b[0])} x {int(b[3] - b[1])} px"})
        if rule:
            facts.insert(0, {"label": "rule", "value": str(rule)})

    return {
        "candidate_id": str(cand.candidate_id), "object_id": str(obj.object_id), "kind": kind,
        "detector": DETECTOR_NAMES.get(kind, kind.replace("_", " ").capitalize()), "rule": rule,
        "score": cand.score, "status": cand.status,
        "proposed": {"class_name": proposed, "class_id": (cand.proposed_label or {}).get("class_id")} if proposed else None,
        "subject": {"object_id": str(obj.object_id), "frame_id": str(obj.frame_id), "class_name": cls,
                    "bbox": _box(obj.bbox), "conf": obj.conf, "state": obj.state, "source": obj.source},
        "frame": {"frame_id": str(obj.frame_id), "width": frame.width if frame else None,
                  "height": frame.height if frame else None, "image_url": f"/api/frames/{obj.frame_id}/image"},
        "reason": reason, "facts": facts, "evidence": evidence, "neighbours": neighbours, "twin_frame": twin,
    }


async def explain_candidate(db: AsyncSession, candidate_id: str) -> dict | None:
    cand = await db.get(ErrorCandidate, uuid.UUID(str(candidate_id)))
    if cand is None:
        return None
    return await explain(db, cand)


async def frame_explanations(db: AsyncSession, frame_id: str, *, status: str | None = "pending",
                             limit: int = 100) -> dict:
    """Every candidate on one frame's objects, explained: what the editor overlay draws."""
    fid = uuid.UUID(str(frame_id))
    frame = await db.get(Frame, fid)
    if frame is None:
        return {"frame_id": str(fid), "width": None, "height": None, "items": []}
    objs = list((await db.execute(select(Object).where(Object.frame_id == fid))).scalars().all())
    by_id = {o.object_id: o for o in objs}
    if not by_id:
        return {"frame_id": str(fid), "width": frame.width, "height": frame.height, "items": []}
    q = select(ErrorCandidate).where(ErrorCandidate.object_id.in_(list(by_id)))
    if status:
        q = q.where(ErrorCandidate.status == status)
    cands = list((await db.execute(q.order_by(ErrorCandidate.score.desc()).limit(limit))).scalars().all())
    names = _Names()
    items = []
    for c in cands:
        e = await explain(db, c, obj=by_id.get(c.object_id), frame=frame, frame_objects=objs, names=names)
        if e is not None:
            items.append(e)
    return {"frame_id": str(fid), "width": frame.width, "height": frame.height, "items": items}
