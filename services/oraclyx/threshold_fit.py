"""Fitting the auto-accept thresholds from measured outcomes, and refusing to fit what cannot be measured.

The gate's thresholds are constants described as precision floors. This reads the outcomes an evaluation
already recorded, fits a per-class operating point against the pack's false-accept bound, and stores it.
The arithmetic is core/accel/np_threshold.py; this module is where the pairs come from and where the
result lands.

WHICH CONFIDENCE. `Prediction` carries `conf` (raw) and `conf_calibrated` (post-isotonic, when a
calibration ran). The fit prefers calibrated and records which it used, because a threshold fitted on one
and applied to the other is not conservative, it is arbitrary: calibration is a monotone remapping, so the
same numeric score means a different probability on each side of it. The gate is told the same field, and
refuses a fit whose field does not match what it is about to threshold.

WHAT COUNTS AS RIGHT. `EvalPatch.outcome` is tp for a matched prediction whose class agreed, fp for a
matched prediction whose class disagreed AND for a prediction that matched nothing at all. An auto-accept
gate is deciding whether to let a label into the corpus unseen, and a box on the right object with the
wrong name is exactly as wrong as a box on nothing, so both count against. The key is `pred_class_id`,
because that is the class the gate would be thresholding when the decision is made.

THE FIT INHERITS THE GOLD SET'S BIAS AND SAYS SO. A false-accept rate here is measured against the sealed
gold denominator, and a prediction on a real object the gold set never labelled is counted as wrong. On
the champion's gold run that is not a small correction: 4,127 predictions above 0.25 are scored against
302 gold objects, so the measured false-accept rate is inflated by exactly the under-labelling that
services/verdyx/blind_audit.py exists to quantify. Fitted thresholds therefore come out far too strict,
and a class that "cannot be auto-accepted at any threshold" may only mean the gold set never recorded what
it was detecting. The caveat travels on the result rather than living in this docstring alone, because a
threshold table is exactly the kind of artifact that gets read without its provenance.

FITTING IS NOT ACTIVATING. Every row is written with `active = false`. This mirrors
services/autolabel/gold_calibrate.py, which fits, reports whether the result is trustworthy, and leaves
switching it on to a person. A fit that silently replaced a live threshold would move the auto-accept
behaviour of the whole engine on the strength of one evaluation, which is precisely the sort of unreviewed
change the constants were at least honest about being.
"""

from __future__ import annotations

import uuid as uuidlib
from uuid import UUID

import numpy as np
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.accel.np_threshold import MIN_SUPPORT, N_BOOTSTRAP, ThresholdEstimate, np_threshold
from core.config import get_settings
from core.logging import get_logger
from db.models import EvalPatch, InferenceRun, Prediction, ThresholdFit

log = get_logger("threshold_fit")

# How wide the bootstrap interval on the threshold may be before the operating point counts as unlocated.
# A threshold known only to within half the score range is not a threshold. Advisory: it is reported on the
# row rather than suppressing it, so a reader can see the fit and see why it should not be switched on.
MAX_INTERVAL_WIDTH = 0.25


async def _pairs_for_run(db: AsyncSession, run: InferenceRun
                         ) -> tuple[dict[int, tuple[list[float], list[bool]]], str]:
    """(score, was-right) per predicted class, and which confidence column the scores came from.

    Only patches carrying a prediction_id are usable: an `fn` patch is a gold object nothing predicted, so
    it has no score to threshold and no bearing on what an accept rule would have done.
    """
    rows = (await db.execute(
        select(EvalPatch.outcome, EvalPatch.pred_class_id, Prediction.conf, Prediction.conf_calibrated)
        .join(Prediction, Prediction.prediction_id == EvalPatch.prediction_id)
        .where(EvalPatch.run_id == run.run_id, EvalPatch.prediction_id.is_not(None),
               EvalPatch.pred_class_id.is_not(None)))).all()

    # Calibrated only if the run actually has it. A partial calibration would mix two score scales inside
    # one class, so the choice is made once for the whole fit and recorded.
    n_cal = sum(1 for _o, _c, _raw, cal in rows if cal is not None)
    use_calibrated = bool(rows) and n_cal == len(rows)
    field = "conf_calibrated" if use_calibrated else "conf"

    out: dict[int, tuple[list[float], list[bool]]] = {}
    for outcome, cid, raw, cal in rows:
        score = cal if use_calibrated else raw
        if score is None:
            # A reconstructed run has null conf: no score means no operating point, and dropping the row
            # is right because there is nothing for a threshold to have done with it.
            continue
        s, m = out.setdefault(int(cid), ([], []))
        s.append(float(score))
        m.append(outcome == "tp")
    return out, field


def _config_threshold(class_name: str) -> float:
    from services.autolabel.ontology import get_ontology
    from services.domain import safety_l1

    cfg = get_settings().gate
    try:
        return (cfg.safety_auto_accept if get_ontology().by_name(class_name).l1 in safety_l1()
                else cfg.auto_accept)
    except Exception:  # noqa: BLE001 - an unknown name is compared against the cautious constant
        return cfg.safety_auto_accept


async def fit_thresholds(db: AsyncSession, *, run_id: str, min_support: int = MIN_SUPPORT,
                         n_boot: int = N_BOOTSTRAP, activate: bool = False) -> dict:
    """Fit one operating point per predicted class from `run_id`'s recorded outcomes.

    `activate` switches the fit into force. It defaults false and the caller has to mean it: this changes
    what the engine accepts without a human looking, across every class at once.
    """
    run = await db.get(InferenceRun, UUID(run_id))
    if run is None:
        return {"error": "inference run not found", "run_id": run_id}
    if (run.params or {}).get("reconstructed"):
        # Its predictions were backfilled from review history and never had a real confidence, so there is
        # no score distribution to place a threshold in.
        return {"error": "this run is reconstructed and has no real confidence distribution",
                "run_id": run_id}

    from packs.registry import default_pack_id, get_pack
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    policy = get_pack(default_pack_id()).safety_policy

    pairs, field = await _pairs_for_run(db, run)
    if not pairs:
        return {"error": "the run has no scored predictions to fit against", "run_id": run_id}

    fit_id = uuidlib.uuid4()
    rows: list[ThresholdFit] = []
    fitted: dict[int, ThresholdEstimate] = {}
    for cid, (scores, matched) in sorted(pairs.items()):
        try:
            name = onto.by_id(cid).name
        except Exception:  # noqa: BLE001 - a class the ontology no longer carries still gets a row
            name = f"class_{cid}"
        alpha = float(policy.accept_far_bound(name))
        est = np_threshold(np.asarray(scores), np.asarray(matched), alpha=alpha,
                           min_support=min_support, n_boot=n_boot)
        fitted[cid] = est
        width = (est.hi - est.lo) if (est.lo is not None and est.hi is not None) else None
        rows.append(ThresholdFit(
            fit_id=fit_id, run_id=run.run_id, model_version=run.model_version, gold_id=run.gold_id,
            class_id=cid, class_name=name, score_field=field, alpha=alpha,
            measured=est.measured,
            reason=(est.reason if est.reason else
                    (f"fitted, but the bootstrap interval spans {width:.3f} of the score range, so the "
                     "operating point is not located well enough to switch on"
                     if width is not None and width > MAX_INTERVAL_WIDTH else None)),
            threshold=est.threshold, threshold_lo=est.lo, threshold_hi=est.hi,
            far_at=est.far_at, accept_rate=est.accept_rate, n_accept=est.n_accept,
            n_pairs=est.n_pairs, n_positive=est.n_positive, n_boot_fit=est.n_boot_fit,
            config_threshold=_config_threshold(name), active=False))
    db.add_all(rows)
    await db.commit()

    if activate:
        await activate_fit(db, str(fit_id))

    n_fit = sum(1 for e in fitted.values() if e.measured)
    n_loose = sum(1 for r in rows if r.measured and r.reason)
    log.info("threshold_fit.fitted", fit=str(fit_id), run=run_id, model=run.model_version,
             score_field=field, classes=len(rows), fitted=n_fit, refused=len(rows) - n_fit,
             unlocated=n_loose, activated=activate)
    return {
        "fit_id": str(fit_id), "run_id": run_id, "model_version": run.model_version,
        "gold_id": run.gold_id, "score_field": field,
        # Travels with every fit. A reader who takes these thresholds at face value will conclude the
        # model is far worse than a blind audit may show it to be.
        "caveat": ("false-accept rates here are measured against the sealed gold denominator, which counts "
                   "a prediction on a real but unlabelled object as wrong; where the gold set under-labels, "
                   "these thresholds are too strict and a class refused at every threshold may only mean "
                   "the gold set never recorded what it was detecting. Compare against a blind audit "
                   "(services/verdyx/blind_audit.py) before treating a refusal as a model defect"),
        "n_classes": len(rows), "n_fitted": n_fit, "n_refused": len(rows) - n_fit,
        "n_unlocated": n_loose, "active": activate,
        "per_class": [{
            "class_id": r.class_id, "class_name": r.class_name, "alpha": r.alpha,
            "measured": r.measured, "reason": r.reason,
            "threshold": r.threshold, "lo": r.threshold_lo, "hi": r.threshold_hi,
            "config_threshold": r.config_threshold,
            "delta": (round(r.threshold - r.config_threshold, 4)
                      if r.threshold is not None and r.config_threshold is not None else None),
            "far_at": r.far_at, "accept_rate": r.accept_rate,
            "n_pairs": r.n_pairs, "n_positive": r.n_positive, "n_accept": r.n_accept,
        } for r in rows],
    }


async def activate_fit(db: AsyncSession, fit_id: str) -> dict:
    """Put a fit into force, retiring whichever fit for the same model was in force before.

    Wholesale, and only within one model version. A threshold set half from one evaluation and half from
    another is not an operating point, and a fit for one model says nothing about another's scores.
    """
    rows = (await db.execute(select(ThresholdFit).where(
        ThresholdFit.fit_id == UUID(fit_id)))).scalars().all()
    if not rows:
        return {"error": "fit not found", "fit_id": fit_id}
    model_version = rows[0].model_version
    await db.execute(update(ThresholdFit)
                     .where(ThresholdFit.model_version == model_version, ThresholdFit.active.is_(True))
                     .values(active=False))
    await db.execute(update(ThresholdFit)
                     .where(ThresholdFit.fit_id == UUID(fit_id)).values(active=True))
    await db.commit()
    log.info("threshold_fit.activated", fit=fit_id, model=model_version, classes=len(rows))
    return {"fit_id": fit_id, "model_version": model_version, "activated": len(rows)}


async def active_thresholds(db: AsyncSession, model_version: str) -> dict:
    """The thresholds in force for a model: {"by_class", "score_field", "fit_id"}, or an empty map.

    Only classes that were actually fitted appear. A class that could not be fitted is deliberately absent
    rather than present with its config value, so the gate falls back explicitly and says so, instead of
    reading a constant that looks like a measurement.
    """
    rows = (await db.execute(select(ThresholdFit).where(
        ThresholdFit.model_version == model_version, ThresholdFit.active.is_(True)))).scalars().all()
    by_class = {r.class_id: float(r.threshold) for r in rows
                if r.measured and r.threshold is not None}
    return {"by_class": by_class,
            "score_field": rows[0].score_field if rows else None,
            "fit_id": str(rows[0].fit_id) if rows else None,
            "n_active": len(rows), "n_usable": len(by_class)}


async def latest_fit(db: AsyncSession, *, model_version: str | None = None) -> dict | None:
    q = select(ThresholdFit.fit_id, ThresholdFit.model_version, ThresholdFit.run_id,
               ThresholdFit.created_at, func.count()).group_by(
        ThresholdFit.fit_id, ThresholdFit.model_version, ThresholdFit.run_id,
        ThresholdFit.created_at).order_by(ThresholdFit.created_at.desc()).limit(1)
    if model_version:
        q = q.where(ThresholdFit.model_version == model_version)
    row = (await db.execute(q)).first()
    if row is None:
        return None
    return {"fit_id": str(row[0]), "model_version": row[1], "run_id": str(row[2]),
            "created_at": row[3].isoformat() if row[3] else None, "n_classes": int(row[4])}


__all__ = ["fit_thresholds", "activate_fit", "active_thresholds", "latest_fit", "MAX_INTERVAL_WIDTH"]
