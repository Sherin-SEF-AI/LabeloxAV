"""Champion/challenger promotion (M4.4). A retrained challenger ships only if it beats the incumbent on
the frozen gold set AND regresses no safety class under Safe-mIoU; otherwise it is discarded and an alert
is recorded. Safety is never automated to zero: a VRU (pedestrian, rider, cyclist, animal) class that
regresses blocks promotion no matter how much overall mAP improved. Applies to both the cloud models and
the edge model (per task). Respects the pause set by drift or the kill switch.

champion_gate is a pure function over the two gold-metric dicts; evaluate_and_promote wires it to the
registry, the governance pause, and the audit log.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import ModelRegistry
from services.autolabel.ontology import get_ontology
from services.govern.audit import record
from services.govern.killswitch import get_state
from services.govern.registry import get_champion, set_champion
from services.recall.gate import resolve_safety_drop, safety_recall_floor, safety_recall_no_regress

log = get_logger("govern_champion")

def _is_safety_class(name: str, onto) -> bool:
    # The safety definition now comes from the active domain pack (AV: l1 in {"vru","animal"}), not a local
    # literal. Byte-identical for AV; a non-AV pack brings its own safety superclasses.
    from services.domain import safety_l1

    try:
        return onto.by_name(name).l1 in safety_l1()
    except Exception:  # noqa: BLE001
        return False


def _map(m: dict) -> float:
    """The AP the gate compares. Hierarchical when it is present, flat otherwise.

    `hier_ap50` is AP at the pack's chosen comparison level (l1 for AV), where a scooter called a
    motorcycle is correct and a scooter called a truck is not. That is the number worth comparing two
    models on: flat AP charges both the same, so it cannot separate two candidates that differ mainly in
    which mistakes they make, which is what a champion and a challenger usually differ in.

    The leaf-level safety floors below are untouched and stay hard. Nothing here lets a model regress a
    safety class by being right about its superclass.
    """
    h = m.get("hier_ap50")
    if h is not None:
        return float(h)
    return float(m.get("map", m.get("map50", 0.0)) or 0.0)


def _recapture(challenger: dict, cfg) -> dict:
    """Whether the gold denominator this gate rests on has been checked against an independent observer.

    Every other floor here (mAP uplift, safety-class AP, safety recall) is computed against the sealed gold
    set, and the gold set was built by people confirming machine boxes far more readily than drawing new
    ones. If that denominator is inflated, the floors are not measuring what their names say, and a
    challenger can clear all of them while being worse at the thing the floors exist to protect.

    A blind audit (services/verdyx/blind_audit.py) estimates the true population by capture-recapture and
    so yields recall against a denominator the model did not help build. Three outcomes:

      unchecked   no scored audit for this run. Advisory unless cfg.blind_audit_required, for the reason
                  given on that setting: a gate that starts red on every promotion gets switched off.
      unmeasured  an audit exists and could not conclude (nothing found by both observers). Never a pass:
                  somebody looked and the answer was that we cannot tell.
      measured    compare gold recall against the estimate. Beyond the tolerance, refuse: at that point the
                  gold metrics are the wrong instrument and the comparison should not be made at all.

    Note which direction the caveat runs. The estimator assumes the two observers are independent, and on
    hard objects they are not, so the estimated population is biased down and the estimated recall up. The
    measured overstatement is therefore itself a LOWER bound on the real one, and refusing on it is the
    conservative call rather than a harsh one.
    """
    est = challenger.get("recapture")
    tol = float(getattr(cfg, "recapture_max_overstatement", 0.15))
    required = bool(getattr(cfg, "blind_audit_required", False))
    if not est:
        return {"ok": not required, "status": "unchecked", "overstatement": None,
                "reasons": ["no blind audit has been scored for this run, so gold recall is unverified "
                            "against an independent observer"
                            + ("; refused (blind_audit_required)" if required
                               else " (advisory: set govern.blind_audit_required to enforce)")]}
    if not est.get("measured"):
        return {"ok": False, "status": "unmeasured", "overstatement": None,
                "reasons": [f"a blind audit was scored and could not conclude: "
                            f"{est.get('reason') or 'no reason recorded'}"]}
    gold_r, model_r = est.get("gold_recall"), est.get("model_recall")
    if gold_r is None or model_r is None:
        return {"ok": False, "status": "unmeasured", "overstatement": None,
                "reasons": ["the blind audit produced no comparable recall pair"]}
    over = round(float(gold_r) - float(model_r), 6)
    if over > tol:
        return {"ok": False, "status": "measured", "overstatement": over,
                "reasons": [f"gold recall overstates measured recall by {over:.3f} (tolerance {tol:.3f}); "
                            f"gold {gold_r:.3f} against {model_r:.3f} estimated from "
                            f"{est.get('n_both')} shared, {est.get('n_model_only')} model-only and "
                            f"{est.get('n_human_only')} human-only objects. Every floor in this gate is "
                            "computed on the gold denominator, so none of them means what it says"]}
    return {"ok": True, "status": "measured", "overstatement": over,
            "reasons": [f"gold recall is within {tol:.3f} of the blind-audit estimate "
                        f"(overstated by {over:.3f})"]}


def champion_gate(challenger: dict, champion: dict | None, onto, cfg, rcfg=None) -> dict:
    """Pure promotion decision. Fail-closed: a challenger that cannot prove its safety (no Safe-mIoU,
    or no safety-class recall) is never promoted, and a safety-class AP, Safe-mIoU, or safety-class
    recall regression always blocks. rcfg is the recall settings (per-class drop tolerance + the recall
    floor); it defaults to the configured phase4.recall."""
    if rcfg is None:
        rcfg = get_settings().phase4.recall

    # Refuse untrustworthy metrics before any comparison. A reconstructed run has no real confidence
    # distribution (its predictions were backfilled from review history), so its AP/PR are not computable and
    # must never gate a promotion. Divergent harnesses (the val-pass and the prediction plane disagreeing on
    # the same gold set beyond tolerance) is a measurement fault, not a result; fail closed on it.
    if challenger.get("reconstructed"):
        return {"promote": False, "beats_map": False, "map_delta": 0.0, "safe_ok": False, "safety_ok": False,
                "regressed_safety": [], "recall_ok": False, "recapture_ok": False,
                "reasons": ["challenger metrics are from a reconstructed run (no real inference); refused"]}
    if challenger.get("harness_divergent"):
        return {"promote": False, "beats_map": False, "map_delta": 0.0, "safe_ok": False, "safety_ok": False,
                "regressed_safety": [], "recall_ok": False, "recapture_ok": False,
                "reasons": ["the val-pass and prediction-plane harnesses diverge beyond tolerance; refused"]}

    map_c, map_ch = _map(challenger), _map(champion or {})
    beats_map = map_c >= map_ch + cfg.min_map_uplift

    # Whether this much evidence can tell the two apart at all.
    #
    # The gate compares point estimates, and on this corpus's gold set a single object moves mAP by roughly
    # ten points: it refused a challenger for "does not beat champion mAP (0.142 vs 0.169)" on nine matched
    # objects. That refusal may be right, but nothing in it could distinguish a real regression from noise.
    #
    # This is advisory on purpose. It never promotes something the floors rejected; it says out loud when a
    # comparison is being made on a sample too small to support it, so the answer to a blocked promotion can
    # be "measure more" rather than "tune the model".
    evidence = _evidence(challenger, champion)

    sm_c, sm_ch = challenger.get("safe_miou"), (champion or {}).get("safe_miou")
    # A missing challenger Safe-mIoU is a fail, not a silent pass: we cannot verify it did not regress
    # safety, so we refuse. With an incumbent baseline present, enforce the max-drop floor.
    if sm_c is None:
        safe_ok = False
    elif sm_ch is None:
        safe_ok = True  # no incumbent baseline to regress against
    else:
        safe_ok = sm_c >= sm_ch - cfg.safe_miou_max_drop

    pc_c = challenger.get("per_class", {}) or {}
    pc_ch = (champion or {}).get("per_class", {}) or {}
    regressed = [cn for cn, ap in pc_ch.items()
                 if _is_safety_class(cn, onto) and pc_c.get(cn, 0.0) < ap - resolve_safety_drop(cn, onto, rcfg)]
    safety_ok = not regressed

    # Recall is the metric the recovery layer exists to fix, so it gates promotion the same way Safe-mIoU
    # does: a safety class below the recall floor, regressed beyond tolerance, or unmeasured, blocks.
    rec_floor = safety_recall_floor(challenger, onto, rcfg)
    rec_reg = safety_recall_no_regress(challenger, champion, onto, rcfg)
    recall_ok = rec_floor["ok"] and rec_reg["ok"]

    # Whether the denominator every floor above is computed against has been checked at all.
    recap = _recapture(challenger, cfg)

    if champion is None:
        # First champion still must clear the safety floor: a present Safe-mIoU and the recall floor. The
        # recapture condition applies here too. A first champion is the model every later comparison is
        # measured against, so letting it in on an unverified denominator would put the bias in the
        # baseline, where nothing downstream could ever detect it.
        promote = bool(sm_c is not None and rec_floor["ok"] and recap["ok"])
        reasons = (["no incumbent; first champion (Safe-mIoU present)"] if sm_c is not None
                   else ["no incumbent but challenger lacks Safe-mIoU; refused (fail-closed)"])
        reasons += rec_floor["reasons"] + recap["reasons"]
        return {"promote": promote, "beats_map": True, "map_delta": round(map_c, 4),
                "safe_ok": sm_c is not None, "safety_ok": True, "regressed_safety": [],
                "recall_ok": rec_floor["ok"], "recapture_ok": recap["ok"],
                "recapture": recap, "reasons": reasons}

    promote = bool(beats_map and safe_ok and safety_ok and recall_ok and recap["ok"])
    reasons: list[str] = []
    if not beats_map:
        reasons.append(f"does not beat champion mAP ({map_c:.3f} vs {map_ch:.3f})")
        if evidence and not evidence["decisive"]:
            reasons.append(f"and the sample cannot separate them: {evidence['detail']}")
    if not safe_ok:
        reasons.append("challenger lacks Safe-mIoU (fail-closed)" if sm_c is None
                       else f"Safe-mIoU regressed ({sm_c} vs {sm_ch})")
    if not safety_ok:
        reasons.append(f"safety-class regression: {regressed}")
    reasons += rec_floor["reasons"] + rec_reg["reasons"] + recap["reasons"]
    if promote:
        reasons.append("beats champion without any safety regression")
    return {"promote": promote, "beats_map": beats_map, "map_delta": round(map_c - map_ch, 4),
            "safe_ok": safe_ok, "safety_ok": safety_ok, "regressed_safety": regressed,
            "recall_ok": recall_ok, "recapture_ok": recap["ok"], "recapture": recap,
            "reasons": reasons, "evidence": evidence}


async def _common_gold_metrics(db, reg, champ, task):
    """Re-score challenger and champion on ONE common sealed gold set (C5). Returns (chal, champ, note): the
    two commensurable metric dicts and an audit note describing which basis was used. Falls back to each
    model's stored metrics (its own val split) only when no gold set or no downloadable weights exist, and
    says so in the note so the comparison basis is never silently misrepresented."""
    from services.govern.gold_eval import evaluate_on_gold, latest_gold_id

    onto = get_ontology()
    gold_id = await latest_gold_id(db, onto.version)
    if gold_id is None:
        return reg.gold_metrics or {}, (champ.gold_metrics if champ else None), "stored_own_split (no gold set sealed)"
    chal_m = await evaluate_on_gold(db, reg.model_version, gold_id)
    champ_m = await evaluate_on_gold(db, champ.model_version, gold_id) if champ else None
    if chal_m is None or (champ is not None and champ_m is None):
        # weights missing for one side: cannot form a fair common comparison, keep stored metrics but flag it
        return (reg.gold_metrics or {}, (champ.gold_metrics if champ else None),
                f"stored_own_split (weights unavailable for common eval on {gold_id})")
    # The blind-audit estimate for the run these metrics came from. champion_gate is pure, so this has to be
    # attached here or the recapture condition would see nothing on every real promotion and fail closed
    # against a check that had in fact been performed.
    chal_m = await _attach_recapture(db, chal_m, gold_id)
    return chal_m, champ_m, f"common_gold:{gold_id}"


async def _attach_recapture(db, metrics: dict, gold_id: str | None) -> dict:
    """Put the scored blind audit for this run onto the metric dict, under "recapture". Absent stays absent.

    Keyed on the prediction-plane run id that produced these metrics, not on the model version: two runs of
    one model at different operating points are different observers, and an audit of one says nothing about
    the other.
    """
    run_id = metrics.get("prediction_plane_run_id")
    if not run_id:
        return metrics
    from services.verdyx.blind_audit import pooled_estimate

    est = await pooled_estimate(db, run_id=str(run_id), gold_id=gold_id)
    return {**metrics, "recapture": est} if est is not None else metrics


async def evaluate_and_promote(db: AsyncSession, challenger_version: str, task: str = "detection") -> dict:
    from sqlalchemy.exc import IntegrityError

    cfg = get_settings().phase4.govern
    onto = get_ontology()
    reg = await db.get(ModelRegistry, challenger_version)
    if reg is None:
        return {"error": "challenger not registered"}
    # Promotion TOCTOU: two concurrent promotions could each read "no better champion" and both call
    # set_champion. The partial unique index uq_model_registry_champion (migration 0060) makes a second
    # champion for the same task impossible: the loser's commit raises IntegrityError instead of silently
    # creating split-brain. Catch it and report a lost race rather than surfacing a 500.
    try:
        return await _evaluate_and_promote_locked(db, reg, challenger_version, task, cfg, onto)
    except IntegrityError:
        await db.rollback()
        log.info("govern.promotion_race_lost", challenger=challenger_version, task=task)
        return {"promoted": False, "race_lost": True,
                "reason": "another promotion for this task committed first"}


async def _evaluate_and_promote_locked(db, reg, challenger_version, task, cfg, onto) -> dict:
    champ = await get_champion(db, task)
    chal_metrics, champ_metrics, basis = await _common_gold_metrics(db, reg, champ, task)
    gate = champion_gate(chal_metrics, champ_metrics, onto, cfg)
    gate["comparison_basis"] = basis

    state = await get_state(db)
    if gate["promote"] and not state.auto_promote_enabled:
        await record(db, "champion", "promotion_paused", challenger_version,
                     {"gate": gate, "paused_reason": state.paused_reason})
        return {"promoted": False, "paused": True, "gate": gate, "reason": state.paused_reason}

    if gate["promote"]:
        prev = champ.model_version if champ else None
        await set_champion(db, challenger_version, task, promoted_from=prev)
        state.champion_version = challenger_version
        await db.commit()
        await record(db, "champion", "promote", challenger_version, {"gate": gate, "promoted_from": prev})
        log.info("govern.promote", challenger=challenger_version, promoted_from=prev)
        from services.integrations.webhooks import emit

        await emit("model.promoted", {"model_version": challenger_version, "task": task,
                                      "promoted_from": prev, "gate": gate})
        _publish_gate(challenger_version, True, gate)
        return {"promoted": True, "champion": challenger_version, "promoted_from": prev, "gate": gate}

    await record(db, "champion", "reject", challenger_version, {"gate": gate})
    log.info("govern.reject", challenger=challenger_version, reasons=gate["reasons"])
    _publish_gate(challenger_version, False, gate)
    return {"promoted": False, "gate": gate, "alert": "challenger rejected: " + "; ".join(gate["reasons"])}


def _publish_gate(model_version: str, promoted: bool, gate: dict) -> None:
    """Record the decision externally, refusals included.

    The refusals are the more useful half of this history. Every model in this corpus was blocked for months,
    first by a measurement fault and then on safety recall floors, and a record that kept only successful
    promotions would show nothing at all for that entire period.
    """
    from services.integrations.mlflow_sink import log_promotion

    log_promotion(model_version=model_version, promoted=promoted, gate=gate)


def _evidence(challenger: dict, champion: dict | None) -> dict | None:
    """Whether the gold set is large enough for the mAP comparison to mean anything.

    Uses recall as the stand-in, because it is the rate that carries a real denominator (gold instances) and
    is already computed as an interval. mAP has no clean binomial n, so pretending to put a Wilson interval on
    it would be worse than admitting the comparison is unquantified.
    """
    if not champion:
        return None
    from services.verdyx.intervals import compare, wilson

    def _rate(m: dict):
        iv = (m.get("intervals") or {}).get("recall")
        if iv and iv.get("n"):
            return wilson(round(iv["value"] * iv["n"]), int(iv["n"]))
        return None

    a, b = _rate(challenger), _rate(champion)
    if a is None or b is None:
        return None
    return compare(a, b)
