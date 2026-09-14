"""Drift detection (M4.4): watch the input embedding distribution, the label distribution, and the
control-sample precision. A breach pauses auto-promotion (a soft pause, not a full kill) and is audited,
so the loop stops shipping models into a world it no longer matches. PSI (population stability index) is
the divergence measure; a control-precision drop below the floor is a direct breach."""

from __future__ import annotations

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import DriftMetric, Frame, FrameEmbedding, Object
from services.govern.audit import record
from services.govern.control_sample import measured_precision
from services.govern.killswitch import pause_auto_promote, resume_auto_promote

log = get_logger("govern_drift")


def psi(ref: list[float], cur: list[float], eps: float = 1e-6) -> float:
    """Population stability index between two binned distributions (proportions over the same bins)."""
    r = np.asarray(ref, dtype=float) + eps
    c = np.asarray(cur, dtype=float) + eps
    r /= r.sum()
    c /= c.sum()
    return float(np.sum((c - r) * np.log(c / r)))


async def _class_hist(db: AsyncSession, session_ids: list[str] | None, n_classes: int) -> np.ndarray:
    q = select(Object.class_id, func.count()).group_by(Object.class_id)
    if session_ids:
        q = q.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id.in_(session_ids))
    hist = np.zeros(n_classes, dtype=float)
    for cid, n in (await db.execute(q)).all():
        if 0 <= cid < n_classes:
            hist[cid] = n
    return hist


def ontology_class_slots() -> int:
    """How many class slots the label histogram must cover: the highest ontology id plus one.

    This was hardcoded to 64 while the ontology carries ~170 classes, so every class with an id at or above
    64 was silently dropped from the histogram and could not contribute to the metric. A distribution shift
    confined to those classes was structurally invisible.
    """
    from services.autolabel.ontology import get_ontology

    ids = [c.id for c in get_ontology().classes]
    return (max(ids) + 1) if ids else 1


async def label_distribution_drift(db: AsyncSession, ref_sessions: list[str], cur_sessions: list[str],
                                   n_classes: int | None = None) -> dict:
    slots = n_classes or ontology_class_slots()
    ref = await _class_hist(db, ref_sessions, slots)
    cur = await _class_hist(db, cur_sessions, slots)
    val = psi(ref.tolist(), cur.tolist())
    breach = val >= get_settings().phase4.govern.drift_psi_breach
    return {"metric": "label_distribution", "value": round(val, 4), "breach": breach, "class_slots": slots}


def projection_axes(dim: int, k: int, seed: int = 20260727) -> np.ndarray:
    """k deterministic random unit vectors in R^dim, as a (k, dim) matrix.

    Drift in a 768-dimensional embedding space cannot be seen through one fixed coordinate: the previous
    implementation projected onto basis vector 0, which measures a single arbitrary feature and is blind to
    a shift in any of the other 767 directions (and to any shift orthogonal to that one axis). Random
    projections are the standard remedy: by Johnson-Lindenstrauss, a distributional difference survives a
    modest number of random projections with high probability, so taking the worst PSI across k axes is a
    sensitive multivariate test that stays cheap and, being seeded, stays reproducible across runs.
    """
    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(k, dim)).astype(np.float32)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return axes


async def _embedding_matrix(db: AsyncSession, session_ids: list[str] | None, limit: int) -> np.ndarray:
    """Sample embeddings for a window. Ordered by frame_id so the sample is a deterministic slice rather
    than whatever Postgres returned first, which made the metric depend on physical row order."""
    q = (select(FrameEmbedding.dino_vec)
         .join(Frame, Frame.frame_id == FrameEmbedding.frame_id)
         .where(FrameEmbedding.dino_vec.isnot(None)))
    if session_ids:
        q = q.where(Frame.session_id.in_(session_ids))
    q = q.order_by(FrameEmbedding.frame_id).limit(limit)
    rows = [np.asarray(v, dtype=np.float32) for v in (await db.execute(q)).scalars()]
    return np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)


def _quantile_bins(ref_vals: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Bin edges from the reference distribution's quantiles.

    Fixed edges over [-1, 1] put nearly every projection value into one or two bins, which drives PSI to ~0
    regardless of the data. Quantile edges spread the reference evenly across bins, which is how PSI is
    conventionally computed and what gives the statistic its sensitivity.
    """
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(ref_vals, qs)
    edges[0], edges[-1] = -np.inf, np.inf
    # collapse duplicate interior edges (a degenerate reference) so np.histogram stays valid
    return np.unique(edges)


async def input_embedding_drift(db: AsyncSession, ref_sessions: list[str], cur_sessions: list[str],
                                k_axes: int = 32, sample_limit: int = 5000) -> dict:
    """Multivariate embedding drift: PSI over k random projections, reported at the worst axis.

    Breaching on the maximum (not the mean) is deliberate: drift concentrated in a few directions is exactly
    the case a mean would wash out, and a drifting subspace is still a drifting world.
    """
    ref_m = await _embedding_matrix(db, ref_sessions, sample_limit)
    cur_m = await _embedding_matrix(db, cur_sessions, sample_limit)
    thresh = get_settings().phase4.govern.drift_psi_breach
    if ref_m.size == 0 or cur_m.size == 0 or ref_m.shape[1] != cur_m.shape[1]:
        # Not enough evidence to make a claim. Report unmeasured rather than a comforting 0.0.
        return {"metric": "input_embedding", "value": 0.0, "breach": False, "measured": False,
                "reason": "no embeddings in one of the windows"}

    axes = projection_axes(ref_m.shape[1], k_axes)
    ref_p, cur_p = ref_m @ axes.T, cur_m @ axes.T          # (n_ref, k), (n_cur, k)
    per_axis: list[float] = []
    for j in range(axes.shape[0]):
        bins = _quantile_bins(ref_p[:, j])
        if len(bins) < 3:                                   # degenerate axis, no usable binning
            continue
        r = np.histogram(ref_p[:, j], bins=bins)[0].astype(float)
        c = np.histogram(cur_p[:, j], bins=bins)[0].astype(float)
        per_axis.append(psi(r.tolist(), c.tolist()))
    if not per_axis:
        return {"metric": "input_embedding", "value": 0.0, "breach": False, "measured": False,
                "reason": "degenerate embedding distribution"}

    worst, mean = float(np.max(per_axis)), float(np.mean(per_axis))
    return {"metric": "input_embedding", "value": round(worst, 4), "breach": worst >= thresh,
            "measured": True, "mean_psi": round(mean, 4), "axes": len(per_axis),
            "n_ref": int(ref_m.shape[0]), "n_cur": int(cur_m.shape[0])}


async def control_precision_drift(db: AsyncSession) -> dict:
    """Measured auto-accept precision against its floor.

    Unmeasured is reported as unmeasured, not as 1.0. It used to substitute a perfect score when no control
    sample had a human verdict, and persist it: with 601 samples waiting and none judged, the drift ledger
    recorded control precision as 100% and a chart of it over time was a flat line at perfect. That is the
    one number in this system a buyer is meant to be able to trust over a self-reported one, and it was
    reporting the absence of evidence as the strongest possible evidence.

    It still does not breach, deliberately. A gate whose precision has never been measured is not the same
    as one measured below its floor, and turning starvation into a breach would pause every promotion on a
    live system for a reason the operator cannot fix in the moment. It is surfaced instead - `unmeasured`
    with the size of the backlog - the same way the champion gate reports a missing safety metric in
    `unchecked` rather than silently passing it.
    """
    floor = get_settings().phase4.govern.control_precision_floor
    prec = await measured_precision(db)
    p = prec["precision"]
    if p is None:
        return {"metric": "control_precision", "value": None, "breach": False, "unmeasured": True,
                "floor": floor, "reviewed": 0, "pending": prec["pending"],
                "detail": (f"{prec['pending']} control samples are waiting for a human verdict; "
                           "the gate's realized precision has never been measured")}
    return {"metric": "control_precision", "value": p, "breach": p < floor, "unmeasured": False,
            "floor": floor, "reviewed": prec["reviewed"], "pending": prec["pending"]}


async def run_drift_scan(db: AsyncSession, ref_sessions: list[str] | None = None,
                         cur_sessions: list[str] | None = None) -> dict:
    """Compute the drift metrics, persist them, and pause auto-promotion on any breach."""
    results = [await control_precision_drift(db)]
    if ref_sessions and cur_sessions:
        results.append(await label_distribution_drift(db, ref_sessions, cur_sessions))
        results.append(await input_embedding_drift(db, ref_sessions, cur_sessions))

    for r in results:
        # An unmeasured metric is not persisted. A gap in the series is honest about there being no
        # measurement; a row carrying a substituted value is a reading that was never taken.
        if r.get("unmeasured") or r["value"] is None:
            log.warning("drift.unmeasured", metric=r["metric"], detail=r.get("detail"))
            continue
        db.add(DriftMetric(metric=r["metric"], window={"ref": ref_sessions, "cur": cur_sessions},
                           value=float(r["value"]), breach=bool(r["breach"])))
    await db.commit()

    breaches = [r for r in results if r["breach"]]
    resumed = False
    if breaches:
        reason = "drift breach: " + ", ".join(f"{b['metric']}={b['value']}" for b in breaches)
        await pause_auto_promote(db, reason)
        await record(db, "drift", "pause_auto_promote", None, {"breaches": breaches})
        log.info("govern.drift_breach", breaches=[b["metric"] for b in breaches])
        from services.integrations.webhooks import emit

        await emit("drift.breached", {"breaches": breaches, "reason": reason})
    else:
        # No breach this scan: lift a prior drift-induced pause so the loop is not trapped forever.
        resumed = await resume_auto_promote(db)
        if resumed:
            await record(db, "drift", "drift_pause_cleared", None,
                         {"note": "auto-promotion stays off; re-enabling it is the operator opt-in"})
            log.info("govern.drift_recovered")
    return {"metrics": results, "breached": [b["metric"] for b in breaches], "paused": bool(breaches),
            "resumed": resumed}
