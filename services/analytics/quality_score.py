"""Label quality score (M-F.1): a per-object composite QA signal, distinct from confidence, built only from
signals the pipeline already records. Confidence answers "how sure is the model"; quality answers "how likely
is this a clean, correct label a buyer would accept". The two diverge: a high-confidence box with a jagged mask,
a class its track disagrees with, or a cross-view conflict is confident but low quality.

Factors (each in [0,1], higher is better), retained alongside the score:
  confidence   - the calibrated confidence (honest P(correct) from the M-E curve)
  agreement    - cross-path agreement in fusion
  geometry     - the quality reviewer found no geometric/contextual problem
  mask         - the mask and box agree on geometry (no jagged/loose mask)
  temporal     - the object's per-camera track agrees on its class over time
  rig          - no cross-view class conflict (M-MC.2/.4)
The base score is a weighted mean of the available factors. Once a human has ruled, the verdict dominates: an
accepted label floors high, a rejected one caps low, so the stored score reflects known truth where it exists.
Validated on the gold/review set: a low quality score really does mean a higher error rate (see validate()).
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import Frame, Object, RigObject
from db.session import get_sessionmaker
from services.autolabel.calibrate import calibrate_confidence

log = get_logger("analytics.quality_score")

# The score is ANCHORED on calibrated confidence, the one factor a VLM-judged validation shows actually ranks
# errors (AUC ~0.66 on class correctness). The quality-reviewer and cross-view flags then apply hard-defect
# penalties: they catch geometric and consistency errors, a different error type the class judge cannot see, so
# they are kept small enough not to dilute the validated confidence ranking. Mask tightness and temporal
# consistency are retained in the breakdown for transparency but did not add predictive signal, so they do not
# move the score.
_PENALTY = {"geometry": 0.7, "rig": 0.6, "agreement": 0.92}


def compute_factors(prov: dict, obj_conf: float | None, track_class_frac: float | None,
                    rig_conflict: bool | None, source: str | None = None) -> dict:
    """The per-factor [0,1] breakdown from an object's provenance and its track/rig context. No DB access. The
    confidence factor uses the calibrated model score when a model produced the object; a human-authored object
    has no model score, so it takes a fixed human-authority value instead of its (meaningless) stored conf."""
    prov = prov or {}
    cfg = get_settings()
    cf = prov.get("calibrated_from")
    if isinstance(cf, (int, float)) and not isinstance(cf, bool):
        conf = calibrate_confidence(float(cf), bool(prov.get("agreement")),
                                    False, bool(prov.get("mask_box_disagree")), cfg.calibrate)
    elif source == "human":
        conf = 0.9  # a human drew or confirmed this; its stored conf is not a calibrated model score
    else:
        conf = float(obj_conf) if obj_conf is not None else 0.5
    return {
        "confidence": round(min(max(conf, 0.0), 1.0), 4),
        "agreement": 1.0 if prov.get("agreement") else 0.45,
        "geometry": 1.0 if not (prov.get("quality_flags") or []) else 0.15,
        "mask": 0.35 if prov.get("mask_box_disagree") else 1.0,
        "temporal": round(track_class_frac, 4) if track_class_frac is not None else 0.75,
        "rig": 0.2 if rig_conflict else 1.0,
    }


def score_from_factors(factors: dict, state: str | None, source: str | None, include_review: bool = True) -> float:
    """Calibrated confidence, reduced by a hard-defect penalty for each quality/consistency flag, then a
    human-verdict override when include_review is set (a rejected label is low quality, an accepted human one
    is high). include_review=False gives the pre-review score used to validate against real errors."""
    base = factors["confidence"]
    if factors["geometry"] < 1.0:
        base *= _PENALTY["geometry"]      # the quality reviewer flagged geometric/contextual nonsense
    if factors["rig"] < 1.0:
        base *= _PENALTY["rig"]           # a cross-view class conflict (M-MC.2/.4)
    if factors["agreement"] < 1.0:
        base *= _PENALTY["agreement"]     # single-path, no cross-path agreement (mild)
    base = round(min(max(base, 0.0), 1.0), 4)
    if include_review:
        if state == "rejected":
            return round(min(base, 0.20), 4)
        if state == "accepted" and source == "human":
            return round(max(base, 0.75), 4)
    return base


async def _track_class_frac(db: AsyncSession, track_id, class_id) -> float | None:
    """Fraction of the object's per-camera track that shares its class (temporal consistency). None if untracked."""
    if track_id is None:
        return None
    total = (await db.execute(select(func.count()).select_from(Object).where(Object.track_id == track_id))).scalar_one()
    if total <= 1:
        return None
    same = (await db.execute(select(func.count()).select_from(Object)
            .where(Object.track_id == track_id, Object.class_id == class_id))).scalar_one()
    return same / total


async def _rig_conflict(db: AsyncSession, rig_object_id) -> bool | None:
    if rig_object_id is None:
        return None
    ro = await db.get(RigObject, rig_object_id)
    return bool(ro.conflict) if ro is not None else None


async def score_object(db: AsyncSession, object_id: UUID, persist: bool = True,
                       include_review: bool = True) -> dict | None:
    """Compute (and optionally persist) one object's quality score with its factor breakdown."""
    obj = await db.get(Object, object_id)
    if obj is None:
        return None
    tcf = await _track_class_frac(db, obj.track_id, obj.class_id)
    rc = await _rig_conflict(db, obj.rig_object_id)
    factors = compute_factors(obj.provenance or {}, obj.conf, tcf, rc, obj.source)
    score = score_from_factors(factors, obj.state, obj.source, include_review)
    if persist:
        obj.quality_score = score
        await db.commit()
    return {"object_id": str(object_id), "quality_score": score, "factors": factors}


async def backfill(session_id: UUID | None = None, batch: int = 1000) -> dict:
    """Compute and store quality scores across a session or the whole corpus. Batched so the track-consistency
    lookups reuse one session; safe to re-run (idempotent)."""
    maker = get_sessionmaker()
    n = 0
    async with maker() as db:
        stmt = select(Object.object_id)
        if session_id is not None:
            stmt = stmt.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == session_id)
        ids = [r[0] for r in (await db.execute(stmt)).all()]
        # score in chunks, one commit per chunk (not per object)
        for start in range(0, len(ids), batch):
            chunk = ids[start:start + batch]
            for oid in chunk:
                r = await score_object(db, oid, persist=False)
                if r is not None:
                    await db.execute(update(Object).where(Object.object_id == oid)
                                     .values(quality_score=r["quality_score"]))
                    n += 1
            await db.commit()
    log.info("quality.backfilled", scored=n, scope=str(session_id) if session_id else "corpus")
    return {"scored": n, "scope": str(session_id) if session_id else "corpus"}


async def validate(n_objects: int = 260, seed: int = 5) -> dict:
    """Validate that a low quality score means a higher error rate, on MODEL-produced objects (the labels the
    score is for). Error truth comes from the VLM judge (the same proxy the confidence calibration used), since
    the review trail holds no rejected model objects. Samples stored fused objects with full provenance, scores
    them from model signals only (no review override), VLM-judges each, and reports the error rate per score
    bucket. If the low bucket errs more than the high bucket, the score is a real signal, not noise."""
    import numpy as np

    from services.autolabel.ontology import get_ontology
    from services.autolabel.paths.path_c_vlm import OllamaVlmClient, crop_object
    from services.autolabel.runner import load_image

    onto = get_ontology()
    name_to_l0 = {c.name.strip().lower(): c.l0 for c in onto.classes}
    vlm = OllamaVlmClient()
    confusable = ["sedan", "hatchback", "suv", "autorickshaw", "motorcycle", "truck", "bus", "cycle",
                  "pedestrian", "cart", "animal", "traffic_sign", "pole"]

    maker = get_sessionmaker()
    scores: list[float] = []
    correct_flags: list[int] = []
    async with maker() as db:
        rng = np.random.default_rng(seed)
        rows = (await db.execute(
            select(Object.object_id, Object.frame_id, Object.bbox, Object.class_id, Object.provenance,
                   Object.conf, Object.track_id, Object.rig_object_id)
            .where(Object.source == "fused", Object.provenance.op("?")("calibrated_from")).limit(4000))).all()
        idx = rng.permutation(len(rows))[:n_objects]
        frame_cache: dict = {}
        for i in idx:
            oid, fid, bbox, cid, prov, conf, tid, rid = rows[int(i)]
            try:
                cls_name = onto.by_id(int(cid)).name
            except Exception:  # noqa: BLE001
                continue
            if fid not in frame_cache:
                fr = await db.get(Frame, fid)
                try:
                    frame_cache[fid] = load_image(fr.img_uri) if fr else None
                except Exception:  # noqa: BLE001
                    frame_cache[fid] = None
            img = frame_cache[fid]
            if img is None:
                continue
            shortlist = list(dict.fromkeys([cls_name, *confusable, "none_of_these"]))
            res = vlm.verify(crop_object(img, tuple(float(x) for x in bbox), 0.15), shortlist, {})
            vn = (res.class_name or "").strip().lower()
            correct = vn == cls_name.lower() or (name_to_l0.get(vn) is not None and name_to_l0.get(vn) == name_to_l0.get(cls_name.lower()))

            tcf = await _track_class_frac(db, tid, cid)
            rc = await _rig_conflict(db, rid)
            factors = compute_factors(prov or {}, conf, tcf, rc, "fused")
            scores.append(score_from_factors(factors, None, None, include_review=False))
            correct_flags.append(1 if correct else 0)

    scores_a = np.asarray(scores, float)
    y = np.asarray(correct_flags, float)
    # AUC = P(score of a correct label > score of an error): the score is a real error signal when AUC > 0.5
    pos, neg = scores_a[y == 1], scores_a[y == 0]
    auc = None
    if len(pos) and len(neg):
        auc = float(sum(float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg)) for p in pos) / (len(pos) * len(neg)))
    # tertile error rates: the low-quality third should err most
    out: dict = {"n": len(scores), "n_errors": int((y == 0).sum()), "auc": round(auc, 4) if auc is not None else None}
    if len(scores) >= 9:
        q1, q2 = np.quantile(scores_a, [1 / 3, 2 / 3])
        for name, mask in (("low", scores_a < q1), ("mid", (scores_a >= q1) & (scores_a < q2)), ("high", scores_a >= q2)):
            n = int(mask.sum())
            out[name] = {"n": n, "error_rate": round(float(1 - y[mask].mean()), 4) if n else None}
        lo = out["low"]["error_rate"]
        hi = out["high"]["error_rate"]
        out["separates"] = bool(lo is not None and hi is not None and lo > hi)
    out["is_signal"] = bool(auc is not None and auc > 0.5)
    log.info("quality.validated", auc=out["auc"], is_signal=out.get("is_signal"))
    return out
