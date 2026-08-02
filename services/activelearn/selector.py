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
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import ErrorCandidate, Object, ObjectEmbedding, ScenarioCandidate
from services.autolabel.gate import is_rare
from services.autolabel.ontology import get_ontology

log = get_logger("al_selector")

# objects worth a human: still provisional (not human-verified), where a label adds signal
_CANDIDATE_STATES = ("review", "annotate", "auto_accept")

# Smallest object a person can actually rule on from a crop, measured on the shorter side in pixels.
#
# The value ranking scores what the model is unsure about, and a distant object is exactly that: small, low
# confidence, high uncertainty. So it sorts to the top of every pool while being the one object a reviewer
# cannot judge. A batch mined without this floor came back at 12 to 40 pixels, which is a queue of coin
# flips: the reviewer guesses, and the guess enters the corpus indistinguishable from a considered verdict.
#
# Zero disables the floor, for callers that are ranking rather than dispatching work.
MIN_REVIEWABLE_SIDE_PX = 0.0

# Frames per track used to estimate flicker.
#
# Flicker is the mean scale-normalised jitter between consecutive boxes, so a bounded prefix estimates it
# from a sample. Fetching every frame of every candidate track was the largest single cost in scoring the
# review queue: 1,155 tracks pulled 127,468 rows with their bboxes, and the median track is 114 frames long.
#
# This is an approximation and it is worth stating what it costs rather than calling it free. Measured
# against the exact computation on the live pool, a prefix reads flicker about 14% high, because a track
# jitters most just after it is initialised, and the per-track ranking is not preserved: at a window of 32,
# 37 of the top 100 flicker tracks change.
#
# What survives is the ranking that is actually handed to a reviewer, because flicker is one of seven terms
# and carries a weight of 0.15. On the final value ordering: window 64 gives Spearman 0.9975 against exact
# with 58 of the top 60 queue positions unchanged, for 6.69s -> 4.56s. Window 32 is faster again at 3.95s
# and drops to 54 of 60, which is a visible change to what somebody is asked to review, so 64 is the trade
# taken.
FLICKER_WINDOW = 64

# What a detector's score is worth before anybody has judged that detector.
#
# Not 1.0, which is the old behaviour and treats an unevaluated detector as reliable. Not 0.0 either, which
# would be defensible in principle ("no evidence it predicts") and useless in practice, because it would
# silence every detector in this corpus at once: 298,529 candidates carry one verdict between them. Half
# says the honest thing, that an unjudged detector is a coin until somebody rules on its candidates, and
# leaves it able to contribute without outranking a detector that has earned its weight.
UNMEASURED_DETECTOR_WEIGHT = 0.5


async def _detector_weights(db) -> dict[str, float]:
    """Per-detector ranking weight: the Wilson lower bound of its measured precision.

    The lower bound rather than the point estimate, because the point estimate rewards small samples: nine
    confirmations out of ten reads as 0.9 and would outrank nine hundred out of a thousand at 0.9, when the
    second is the one that has actually been demonstrated. The lower bound puts them at roughly 0.60 and
    0.88, which is the order a reviewer would want.
    """
    from services.errordetect.queue import MIN_VERDICTS_FOR_PRECISION, detector_precision

    report = await detector_precision(db)
    out: dict[str, float] = {}
    for kind, d in report["per_kind"].items():
        if d["decided"] >= MIN_VERDICTS_FOR_PRECISION:
            out[kind] = float(d["precision"]["lo"])
    return out


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


async def score_candidates(db: AsyncSession, session_id: str | None = None, pool_limit: int = 2000,
                           class_ids: list[int] | None = None,
                           min_side_px: float = MIN_REVIEWABLE_SIDE_PX) -> list[dict]:
    cfg = get_settings().phase4.activelearn
    onto = get_ontology()

    q = (select(Object.object_id, Object.frame_id, Object.class_id, Object.conf, Object.provenance,
                Object.quality_score, Object.track_id)
         .where(Object.state.in_(_CANDIDATE_STATES), Object.source != "human"))
    if min_side_px > 0:
        # Applied in the pool query rather than after ranking, because the pool is itself truncated by
        # pool_limit. Filtering afterwards would let unreviewable objects consume the pool and leave the
        # ranking to choose from whatever few judgeable ones happened to survive.
        q = q.where(func.least(Object.bbox[3] - Object.bbox[1],
                               Object.bbox[4] - Object.bbox[2]) >= min_side_px)
    if session_id:
        from db.models import Frame
        q = q.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == session_id)
    if class_ids:
        # class-focused mining: scope the pool to specific ontology classes (e.g. safety classes the
        # champion gate is blocked on) so the value ranking is computed within that class set.
        q = q.where(Object.class_id.in_(class_ids))
    # Order before limiting. An unordered LIMIT hands back whatever Postgres reads first, so the "pool" was an
    # arbitrary slice of a corpus far larger than pool_limit and every downstream value ranking inherited that
    # bias, silently and irreproducibly. Ordering by confidence distance from the decision boundary keeps the
    # objects a value ranking is most likely to care about, and ties break on object_id so the pool is stable
    # across runs rather than shifting with physical row order.
    q = q.order_by(func.abs(Object.conf - 0.5), Object.object_id).limit(pool_limit)
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
    # Error-candidate signal, weighted by how much each detector has earned.
    #
    # This used to be a bare max() over the raw scores, which assumes the detectors emit commensurable
    # numbers. They do not. `confident_learning` reports an actual probability; `policy_violation` and
    # `critic_flag` use hand-assigned constants; `near_dup_inconsistent` was reporting frame similarity,
    # which could not fall below its own 0.96 gate and so beat everything else on every object it touched.
    # A max over that is a max over which detector is loudest.
    #
    # Weighting by measured precision fixes the comparison and closes the loop at the same time: judging a
    # detector in the error queue now changes how the selector ranks. The weight is the Wilson lower bound
    # rather than the point estimate, so nine confirmations out of ten does not outrank nine hundred out of
    # a thousand on the strength of a small sample.
    weights = await _detector_weights(db)
    err_scores: dict[str, float] = {}
    for oid, kind, sc in (await db.execute(
            select(ErrorCandidate.object_id, ErrorCandidate.kind, ErrorCandidate.score).where(
                ErrorCandidate.object_id.in_(oids), ErrorCandidate.status == "pending"))).all():
        weighted = float(sc) * weights.get(kind, UNMEASURED_DETECTOR_WEIGHT)
        err_scores[str(oid)] = max(err_scores.get(str(oid), 0.0), weighted)

    # novelty: mean cosine distance to the k nearest neighbours in the pool (isolated = novel)
    novelty = _pool_novelty([emb.get(oid) for oid in oids], cfg.diversity_knn)

    # temporal flicker per track: a box that jumps or breathes frame-to-frame is a likely auto-label failure
    track_ids = {r[6] for r in rows if r[6] is not None}
    flicker_map = await _track_flicker(db, track_ids)

    items = []
    for i, (oid, fid, cid, conf, prov, qscore, tid) in enumerate(rows):
        prov = prov or {}
        u = _uncertainty(float(conf or 0.0), bool(prov.get("agreement", True)),
                         bool(prov.get("mask_box_disagree", False)), cfg.uncertainty_lo, cfg.uncertainty_hi)
        # ensemble class-distribution entropy (persisted at fusion): a split-vote object reads uncertain even
        # when its scalar confidence sits outside the informative band. Squash nats into [0, 1).
        en = prov.get("entropy")
        if en:
            u = max(u, 1.0 - float(np.exp(-float(en))))
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

        # What the reasoning layer thought of this object, which until now the queue could not see.
        #
        # Conflict is the useful part rather than the score. A low score means the evidence agrees the label
        # is wrong, and the gate already routes those. Conflict means the evidence disagrees with itself:
        # the detector is confident and physics says impossible, or every path proposed a different class.
        # Those are the objects where a human adds the most, because the machine genuinely cannot settle it,
        # and no other term in this ranking can see them.
        #
        # Measured, not assumed: the checks feeding this carry lift over the review base rate (elevation
        # 1.60, temporal 1.25, cross_model 1.23), which is more than can be said for error_prone, currently
        # firing on 40% of the corpus at a near-constant score.
        reasoning = prov.get("reasoning") or {}
        conflict = float(reasoning.get("conflict") or 0.0)
        # An adjudicate verdict is the layer saying out loud that it could not decide, so it counts even
        # when the numeric conflict is modest.
        if str(reasoning.get("decision")) == "adjudicate":
            conflict = max(conflict, 0.5)
        items.append({"object_id": str(oid), "frame_id": str(fid), "class_id": cid,
                      "class_name": onto.by_id(cid).name, "conf": float(conf or 0.0),
                      "quality_score": float(qscore) if qscore is not None else None,
                      "_u": u, "_r": rare, "_n": float(novelty[i]), "_e": err_scores.get(str(oid), 0.0),
                      "_fl": flicker_map.get(tid, 0.0), "_f": fn, "_rc": conflict})

    u = _norm([it["_u"] for it in items])
    r = _norm([it["_r"] for it in items])
    n = _norm([it["_n"] for it in items])
    e = _norm([it["_e"] for it in items])
    fl = _norm([it["_fl"] for it in items])
    f = _norm([it["_f"] for it in items])
    rcf = _norm([it["_rc"] for it in items])
    for i, it in enumerate(items):
        it["scores"] = {"uncertainty": round(float(u[i]), 4), "diversity": round(float(n[i]), 4),
                        "rarity": round(float(r[i]), 4), "error_prone": round(float(e[i]), 4),
                        "flicker": round(float(fl[i]), 4), "fn": round(float(f[i]), 4),
                        "reasoner_conflict": round(float(rcf[i]), 4)}
        it["value"] = round(float(cfg.w_uncertainty * u[i] + cfg.w_diversity * n[i]
                                  + cfg.w_rarity * r[i] + cfg.w_error_prone * e[i]
                                  + cfg.w_flicker * fl[i] + cfg.w_fn * f[i]
                                  + cfg.w_reasoner_conflict * rcf[i]), 5)
        for k in ("_u", "_r", "_n", "_e", "_fl", "_f", "_rc"):
            it.pop(k)
    items.sort(key=lambda x: x["value"], reverse=True)
    log.info("al.scored", pool=len(items), session_id=session_id)
    return items


async def _track_flicker(db: AsyncSession, track_ids: set) -> dict:
    """Per-track temporal flicker (scale-normalized box jitter across consecutive frames) for the candidate
    tracks, via core.accel.uncertainty.flicker_scores. A high-flicker track is a likely auto-label failure the
    scalar confidence misses. Tracks with a single frame flicker 0. Returns {track_id: flicker}."""
    if not track_ids:
        return {}
    from sqlalchemy import func as _func

    from core.accel.uncertainty import flicker_scores
    from db.models import Frame

    # A window per track rather than the whole track.
    #
    # Flicker is the mean scale-normalised jitter between consecutive boxes, so it is an average over
    # differences and a bounded prefix estimates it as well as the full sequence does. Fetching everything
    # was the single largest cost in scoring the review queue: 1,155 candidate tracks pulled 127,468 rows
    # with their bboxes, which is three seconds of pure transfer on a page that opens every session, and the
    # median track is 114 frames long so almost all of it was redundant.
    ranked = (
        select(Object.track_id.label("tid"), Frame.ts_ns.label("ts"),
               Object.bbox.label("bbox"), Object.conf.label("conf"),
               _func.row_number().over(partition_by=Object.track_id,
                                       order_by=Frame.ts_ns).label("rn"))
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(Object.track_id.in_(track_ids))
        .subquery()
    )
    rows = (await db.execute(
        select(ranked.c.tid, ranked.c.ts, ranked.c.bbox, ranked.c.conf)
        .where(ranked.c.rn <= FLICKER_WINDOW)
        .order_by(ranked.c.tid, ranked.c.ts))).all()
    seqs: dict = {}
    for tid, _ts, bbox, conf in rows:
        seqs.setdefault(tid, []).append((bbox, conf))
    tids = list(seqs)
    if not tids:
        return {}
    T = max(len(seqs[t]) for t in tids)
    M = len(tids)
    boxes = np.zeros((M, T, 4))
    confs = np.zeros((M, T))
    valid = np.zeros((M, T), dtype=bool)
    for m, t in enumerate(tids):
        for k, (bbox, conf) in enumerate(seqs[t]):
            boxes[m, k] = bbox
            confs[m, k] = float(conf or 0.0)
            valid[m, k] = True
    out = flicker_scores(boxes, confs, valid)
    return {tids[m]: float(out["flicker"][m]) for m in range(M)}


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
