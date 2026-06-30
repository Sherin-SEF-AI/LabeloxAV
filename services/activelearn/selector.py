"""Active-learning value scoring (M4.0). Rank unlabeled and low-confidence objects by how much labeling
them would improve the model, combining four signals that already exist in the system:

  uncertainty   - M9 calibrated confidence near the gate boundary + path disagreement (provenance)
  diversity     - DINOv3 embedding novelty (isolated in the pool, not a near-duplicate of labeled data)
  rarity        - rare/fallback classes (is_rare) + Phase 1 rare-scenario frames (scenario_candidate)
  error_prone   - the M4.1 error candidates (objects already suspected wrong)

Each signal is min-max normalized across the candidate pool so the configured weights are comparable.
This is pure scoring over existing probabilities and embeddings; it runs locally.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import ErrorCandidate, Object, ObjectEmbedding, ScenarioCandidate
from services.autolabel.gate import is_rare
from services.autolabel.ontology import get_ontology

log = get_logger("al_selector")

# objects worth a human: still provisional (not human-verified), where a label adds signal
_CANDIDATE_STATES = ("review", "annotate", "auto_accept")


def _norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def _uncertainty(conf: float, agreement: bool, mask_box_disagree: bool, lo: float, hi: float) -> float:
    """Peaks in the informative confidence band, boosted by path disagreement."""
    mid, half = (lo + hi) / 2.0, max((hi - lo) / 2.0, 1e-6)
    band = float(np.exp(-(((conf - mid) / half) ** 2)))  # gaussian over the band
    bonus = (0.0 if agreement else 0.35) + (0.2 if mask_box_disagree else 0.0)
    return min(1.0, band + bonus)


async def score_candidates(db: AsyncSession, session_id: str | None = None, pool_limit: int = 2000) -> list[dict]:
    cfg = get_settings().phase4.activelearn
    onto = get_ontology()

    q = (select(Object.object_id, Object.frame_id, Object.class_id, Object.conf, Object.provenance)
         .where(Object.state.in_(_CANDIDATE_STATES), Object.source != "human"))
    if session_id:
        from db.models import Frame
        q = q.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == session_id)
    q = q.limit(pool_limit)
    rows = (await db.execute(q)).all()
    if not rows:
        return []

    oids = [r[0] for r in rows]
    # class frequency over the corpus' accepted labels, for inverse-frequency rarity
    class_counts: dict[int, int] = {}
    for cid in (await db.execute(select(Object.class_id).where(Object.state.in_(("accepted", "auto_accept"))))).scalars():
        class_counts[cid] = class_counts.get(cid, 0) + 1
    max_count = max(class_counts.values()) if class_counts else 1

    # embeddings for novelty (diversity); rows without an embedding get median novelty
    emb_rows = (await db.execute(
        select(ObjectEmbedding.object_id, ObjectEmbedding.dino_vec).where(ObjectEmbedding.object_id.in_(oids)))).all()
    emb = {oid: np.asarray(v, dtype=np.float32) for oid, v in emb_rows}

    # rare-scenario frames (Phase 1 discovery), and error candidates (M4.1)
    rare_frames = set((await db.execute(
        select(ScenarioCandidate.frame_id).where(ScenarioCandidate.kind.in_(("rare_class", "embedding_outlier"))))).scalars())
    err_scores: dict[str, float] = {}
    for oid, sc in (await db.execute(
            select(ErrorCandidate.object_id, ErrorCandidate.score).where(
                ErrorCandidate.object_id.in_(oids), ErrorCandidate.status == "pending"))).all():
        err_scores[str(oid)] = max(err_scores.get(str(oid), 0.0), float(sc))

    # novelty: mean cosine distance to the k nearest neighbours in the pool (isolated = novel)
    novelty = _pool_novelty([emb.get(oid) for oid in oids], cfg.diversity_knn)

    items = []
    for i, (oid, fid, cid, conf, prov) in enumerate(rows):
        prov = prov or {}
        u = _uncertainty(float(conf or 0.0), bool(prov.get("agreement", True)),
                         bool(prov.get("mask_box_disagree", False)), cfg.uncertainty_lo, cfg.uncertainty_hi)
        rare = 0.6 if is_rare(cid, onto) else 0.0
        rare = max(rare, 1.0 - class_counts.get(cid, 0) / max_count)
        if fid in rare_frames:
            rare = min(1.0, rare + 0.25)
        # recall-recovery value: a recovered miss carries its fn_value in provenance; this term only
        # orders the pool so a trackgap recovery outranks a speculative region crop (it already entered
        # the pool via source != "human" and state="review").
        # raw_conf is a dict for recall-recovered objects but a bare scalar for older/imported ones, so
        # only read fn_value when it is actually a dict (else this term is 0).
        rc = prov.get("raw_conf")
        fn = float(rc.get("fn_value", 0.0)) if isinstance(rc, dict) else 0.0
        items.append({"object_id": str(oid), "frame_id": str(fid), "class_id": cid,
                      "class_name": onto.by_id(cid).name, "conf": float(conf or 0.0),
                      "_u": u, "_r": rare, "_n": float(novelty[i]), "_e": err_scores.get(str(oid), 0.0), "_f": fn})

    u = _norm([it["_u"] for it in items])
    r = _norm([it["_r"] for it in items])
    n = _norm([it["_n"] for it in items])
    e = _norm([it["_e"] for it in items])
    f = _norm([it["_f"] for it in items])
    for i, it in enumerate(items):
        it["scores"] = {"uncertainty": round(float(u[i]), 4), "diversity": round(float(n[i]), 4),
                        "rarity": round(float(r[i]), 4), "error_prone": round(float(e[i]), 4),
                        "fn": round(float(f[i]), 4)}
        it["value"] = round(float(cfg.w_uncertainty * u[i] + cfg.w_diversity * n[i]
                                  + cfg.w_rarity * r[i] + cfg.w_error_prone * e[i] + cfg.w_fn * f[i]), 5)
        for k in ("_u", "_r", "_n", "_e", "_f"):
            it.pop(k)
    items.sort(key=lambda x: x["value"], reverse=True)
    log.info("al.scored", pool=len(items), session_id=session_id)
    return items


def _pool_novelty(vecs: list[np.ndarray | None], k: int) -> np.ndarray:
    present = [(i, v) for i, v in enumerate(vecs) if v is not None]
    out = np.full(len(vecs), 0.5, dtype=float)  # default for embeddingless rows
    if len(present) < 2:
        return out
    idx = [i for i, _ in present]
    mat = np.stack([v / (np.linalg.norm(v) + 1e-9) for _, v in present])
    sim = mat @ mat.T
    np.fill_diagonal(sim, -1.0)
    kk = min(k, sim.shape[0] - 1)
    topk = np.sort(sim, axis=1)[:, -kk:]
    nov = 1.0 - topk.mean(axis=1)  # far from neighbours = novel
    for j, i in enumerate(idx):
        out[i] = float(nov[j])
    return out
