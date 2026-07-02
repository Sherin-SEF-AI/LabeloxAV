"""The Ontology Steward: automates the M-Q.0 promotion case so the ontology grows from a reviewed pipeline
instead of ad-hoc governance.

It watches the fallback buckets (vehicle_fallback / object_fallback), clusters them by appearance (DINOv3),
and when a cluster grows past the promotion threshold it assembles an evidence packet -- the instance count,
a visual crop grid, the visually-nearest existing classes (the confusion risks and a naming hint) -- and
files a PromotionProposal awaiting a one-click approve or reject. Approval mints the class and relabels the
cluster as one reversible run; the steward proposes, a human disposes. This is the answer to getting from
45 classes to 150 without repeating the bus_shelter mistake.
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import AgentRun, Frame, Object, ObjectEmbedding, PromotionProposal

log = get_logger("agent.ontology_steward")

_KIND = "ontology_promotion"


def _normed(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _cluster(rows: list[tuple[str, object]], sim_thresh: float) -> list[dict]:
    """Greedy online cosine clustering: assign each fallback crop to the nearest centroid above the
    threshold, else open a new cluster. O(n*k), fine for a bounded sample."""
    clusters: list[dict] = []
    for oid, vec in rows:
        v = _normed(vec)
        best_s, best_i = sim_thresh, -1
        for i, c in enumerate(clusters):
            s = float(c["vec"] @ v)
            if s > best_s:
                best_s, best_i = s, i
        if best_i >= 0:
            c = clusters[best_i]
            n = len(c["members"])
            c["vec"] = _normed(c["vec"] * n + v)
            c["members"].append(str(oid))
        else:
            clusters.append({"vec": v, "members": [str(oid)]})
    return clusters


async def _confusion_and_hint(db: AsyncSession, centroid: np.ndarray, onto) -> tuple[list[dict], str | None]:
    """The visually-nearest existing (non-fallback) classes to the cluster centroid: confusion risks, and the
    top one as a naming hint."""
    from core.embeddings import object_neighbors

    neigh = await object_neighbors(db, centroid.tolist(), k=40)
    if not neigh:
        return [], None
    oids = [uuid.UUID(o) for o, _ in neigh]
    rows = (await db.execute(select(Object.object_id, Object.class_id).where(Object.object_id.in_(oids)))).all()
    cls_of = {str(oid): cid for oid, cid in rows}
    counts: Counter = Counter()
    for oid, _sim in neigh:
        cid = cls_of.get(oid)
        if cid is None or onto.is_fallback(int(cid)):
            continue
        try:
            counts[onto.by_id(int(cid)).name] += 1
        except Exception:  # noqa: BLE001
            continue
    total = sum(counts.values()) or 1
    confusion = [{"class": name, "share": round(n / total, 3)} for name, n in counts.most_common(5)]
    return confusion, (confusion[0]["class"] if confusion else None)


async def _crop_grid(db: AsyncSession, object_ids: list[str], proposal_id: uuid.UUID) -> str | None:
    """Tile up to 16 member crops into one evidence image in the object store."""
    import cv2

    from services.autolabel.paths.path_c_qwen3vl import crop_object
    from services.recall.backends import load_image_bgr

    store = get_object_store()
    rows = (await db.execute(
        select(Object.bbox, Frame.img_uri).join(Frame, Frame.frame_id == Object.frame_id)
        .where(Object.object_id.in_([uuid.UUID(o) for o in object_ids[:16]])))).all()
    tiles = []
    for bbox, uri in rows:
        try:
            crop = crop_object(load_image_bgr(store, uri), tuple(float(x) for x in bbox), 0.1)
            tiles.append(cv2.resize(crop, (96, 96)))
        except Exception:  # noqa: BLE001
            continue
    if not tiles:
        return None
    while len(tiles) < 16:
        tiles.append(np.zeros((96, 96, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)])
    ok, buf = cv2.imencode(".png", grid)
    if not ok:
        return None
    return store.put_bytes(f"ontology_evidence/{proposal_id}.png", buf.tobytes(), "image/png")


async def scan_fallbacks(db: AsyncSession, *, sample: int = 3000, min_cluster: int = 30,
                         sim_thresh: float = 0.62, max_members: int = 500) -> dict:
    """Cluster the fallback buckets and file a PromotionProposal for every cluster past the threshold. A fresh
    scan supersedes prior undecided proposals (approved/rejected are kept)."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    fb = onto.fallback_ids()
    rows = (await db.execute(
        select(ObjectEmbedding.object_id, ObjectEmbedding.dino_vec)
        .join(Object, Object.object_id == ObjectEmbedding.object_id)
        .where(Object.class_id.in_(fb), Object.source != "human")
        .limit(sample))).all()
    if not rows:
        return {"scanned": 0, "clusters": 0, "proposals": 0}

    # infer each cluster's parent fallback class from its members (majority)
    oid_class = dict((str(oid), cid) for oid, cid in (await db.execute(
        select(Object.object_id, Object.class_id).where(
            Object.object_id.in_([o for o, _ in rows])))).all())

    clusters = [c for c in _cluster(rows, sim_thresh) if len(c["members"]) >= min_cluster]

    await db.execute(delete(PromotionProposal).where(PromotionProposal.status == "proposed"))
    made = 0
    for c in clusters:
        confusion, hint = await _confusion_and_hint(db, c["vec"], onto)
        from_class = Counter(oid_class.get(m) for m in c["members"]).most_common(1)[0][0] or fb[0]
        pid = uuid.uuid4()
        evidence = await _crop_grid(db, c["members"], pid)
        db.add(PromotionProposal(
            proposal_id=pid, from_class=int(from_class), member_count=len(c["members"]),
            rep_object_ids=c["members"][:max_members], suggested_name=hint, confusion_classes=confusion,
            evidence_uri=evidence, status="proposed"))
        made += 1
    await db.commit()
    log.info("steward.scan", scanned=len(rows), clusters=len(clusters), proposals=made)
    return {"scanned": len(rows), "clusters": len(clusters), "proposals": made}


async def approve(db: AsyncSession, proposal_id: uuid.UUID, name: str, *, l0: str = "object",
                  l1: str = "custom", created_by: str | None = None) -> dict:
    """Mint the class and relabel the cluster to it as one reversible run."""
    from services.autolabel.ontology import add_custom_class

    prop = await db.get(PromotionProposal, proposal_id)
    if prop is None:
        raise ValueError("proposal not found")
    if prop.status != "proposed":
        raise ValueError(f"proposal is {prop.status}")
    cls = add_custom_class(name, l0=l0, l1=l1)
    new_id = int(cls["id"])
    # Mirror the new class into the ontology_class table so the object.class_id FK accepts it (same step the
    # meta router does when an annotator adds a class by hand).
    from db.models import OntologyClass
    from services.autolabel.ontology import get_ontology

    if (await db.execute(select(OntologyClass.id).where(OntologyClass.id == new_id).limit(1))).first() is None:
        db.add(OntologyClass(id=new_id, version=get_ontology().version, name=cls["name"], l0=cls["l0"],
                             l1=cls["l1"], india=cls.get("india", True), map_to={}))
        await db.flush()

    run_id = uuid.uuid4()
    changes: dict = {}
    for oid in prop.rep_object_ids:
        obj = await db.get(Object, uuid.UUID(oid))
        if obj is None or obj.source == "human" or int(obj.class_id) == new_id:
            continue
        changes[oid] = {"from_class": int(obj.class_id)}
        obj.class_id = new_id
        obj.version = (obj.version or 0) + 1
        prov = dict(obj.provenance or {})
        prov["agent_run_id"] = str(run_id)
        obj.provenance = prov
    db.add(AgentRun(run_id=run_id, kind=_KIND, scope={"proposal_id": str(proposal_id)}, status="committed",
                    policy={"name": name}, counts={"relabeled": len(changes)}, changes=changes, critic={},
                    created_by=created_by or "ontology_steward"))
    prop.status = "approved"
    prop.approved_class = new_id
    prop.run_id = run_id
    prop.decided_at = datetime.now(timezone.utc)
    await db.commit()
    log.info("steward.approve", proposal_id=str(proposal_id), new_class=new_id, relabeled=len(changes))
    return {"proposal_id": str(proposal_id), "class_id": new_id, "name": cls["name"], "relabeled": len(changes),
            "run_id": str(run_id)}


async def reject(db: AsyncSession, proposal_id: uuid.UUID) -> dict:
    prop = await db.get(PromotionProposal, proposal_id)
    if prop is None:
        raise ValueError("proposal not found")
    prop.status = "rejected"
    prop.decided_at = datetime.now(timezone.utc)
    await db.commit()
    return {"proposal_id": str(proposal_id), "status": "rejected"}


async def list_proposals(db: AsyncSession, status: str = "proposed", limit: int = 50) -> list[dict]:
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    rows = (await db.execute(select(PromotionProposal).where(PromotionProposal.status == status)
                             .order_by(PromotionProposal.member_count.desc()).limit(limit))).scalars().all()

    def _name(cid):
        try:
            return onto.by_id(int(cid)).name
        except Exception:  # noqa: BLE001
            return str(cid)

    return [{"proposal_id": str(p.proposal_id), "from_class": _name(p.from_class), "member_count": p.member_count,
             "suggested_name": p.suggested_name, "confusion_classes": p.confusion_classes,
             "evidence_uri": p.evidence_uri, "sample_object_ids": (p.rep_object_ids or [])[:8],
             "status": p.status} for p in rows]
