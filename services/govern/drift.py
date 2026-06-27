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
from services.govern.killswitch import pause_auto_promote

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


async def label_distribution_drift(db: AsyncSession, ref_sessions: list[str], cur_sessions: list[str],
                                   n_classes: int = 64) -> dict:
    ref = await _class_hist(db, ref_sessions, n_classes)
    cur = await _class_hist(db, cur_sessions, n_classes)
    val = psi(ref.tolist(), cur.tolist())
    breach = val >= get_settings().phase4.govern.drift_psi_breach
    return {"metric": "label_distribution", "value": round(val, 4), "breach": breach}


async def _embedding_proj_hist(db: AsyncSession, session_ids: list[str] | None, axis: np.ndarray,
                               bins: np.ndarray) -> np.ndarray:
    q = select(FrameEmbedding.dino_vec).join(Frame, Frame.frame_id == FrameEmbedding.frame_id).where(
        FrameEmbedding.dino_vec.isnot(None))
    if session_ids:
        q = q.where(Frame.session_id.in_(session_ids))
    vals = [float(np.asarray(v, dtype=np.float32) @ axis) for v in (await db.execute(q.limit(5000))).scalars()]
    if not vals:
        return np.zeros(len(bins) - 1)
    return np.histogram(vals, bins=bins)[0].astype(float)


async def input_embedding_drift(db: AsyncSession, ref_sessions: list[str], cur_sessions: list[str]) -> dict:
    axis = np.zeros(768, dtype=np.float32)
    axis[0] = 1.0  # a fixed projection axis keeps the metric stable across runs
    bins = np.linspace(-1.0, 1.0, 11)
    ref = await _embedding_proj_hist(db, ref_sessions, axis, bins)
    cur = await _embedding_proj_hist(db, cur_sessions, axis, bins)
    val = psi(ref.tolist(), cur.tolist())
    breach = val >= get_settings().phase4.govern.drift_psi_breach
    return {"metric": "input_embedding", "value": round(val, 4), "breach": breach}


async def control_precision_drift(db: AsyncSession) -> dict:
    floor = get_settings().phase4.govern.control_precision_floor
    prec = await measured_precision(db)
    p = prec["precision"]
    breach = p is not None and p < floor
    return {"metric": "control_precision", "value": p if p is not None else 1.0, "breach": breach,
            "floor": floor, "reviewed": prec["reviewed"]}


async def run_drift_scan(db: AsyncSession, ref_sessions: list[str] | None = None,
                         cur_sessions: list[str] | None = None) -> dict:
    """Compute the drift metrics, persist them, and pause auto-promotion on any breach."""
    results = [await control_precision_drift(db)]
    if ref_sessions and cur_sessions:
        results.append(await label_distribution_drift(db, ref_sessions, cur_sessions))
        results.append(await input_embedding_drift(db, ref_sessions, cur_sessions))

    for r in results:
        db.add(DriftMetric(metric=r["metric"], window={"ref": ref_sessions, "cur": cur_sessions},
                           value=float(r["value"]), breach=bool(r["breach"])))
    await db.commit()

    breaches = [r for r in results if r["breach"]]
    if breaches:
        reason = "drift breach: " + ", ".join(f"{b['metric']}={b['value']}" for b in breaches)
        await pause_auto_promote(db, reason)
        await record(db, "drift", "pause_auto_promote", None, {"breaches": breaches})
        log.info("govern.drift_breach", breaches=[b["metric"] for b in breaches])
    return {"metrics": results, "breached": [b["metric"] for b in breaches], "paused": bool(breaches)}
