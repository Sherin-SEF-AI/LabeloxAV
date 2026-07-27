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
    return float(m.get("map", m.get("map50", 0.0)) or 0.0)


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
                "regressed_safety": [], "recall_ok": False,
                "reasons": ["challenger metrics are from a reconstructed run (no real inference); refused"]}
    if challenger.get("harness_divergent"):
        return {"promote": False, "beats_map": False, "map_delta": 0.0, "safe_ok": False, "safety_ok": False,
                "regressed_safety": [], "recall_ok": False,
                "reasons": ["the val-pass and prediction-plane harnesses diverge beyond tolerance; refused"]}

    map_c, map_ch = _map(challenger), _map(champion or {})
    beats_map = map_c >= map_ch + cfg.min_map_uplift

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

    if champion is None:
        # First champion still must clear the safety floor: a present Safe-mIoU and the recall floor.
        promote = bool(sm_c is not None and rec_floor["ok"])
        reasons = (["no incumbent; first champion (Safe-mIoU present)"] if sm_c is not None
                   else ["no incumbent but challenger lacks Safe-mIoU; refused (fail-closed)"])
        reasons += rec_floor["reasons"]
        return {"promote": promote, "beats_map": True, "map_delta": round(map_c, 4),
                "safe_ok": sm_c is not None, "safety_ok": True, "regressed_safety": [],
                "recall_ok": rec_floor["ok"], "reasons": reasons}

    promote = bool(beats_map and safe_ok and safety_ok and recall_ok)
    reasons: list[str] = []
    if not beats_map:
        reasons.append(f"does not beat champion mAP ({map_c:.3f} vs {map_ch:.3f})")
    if not safe_ok:
        reasons.append("challenger lacks Safe-mIoU (fail-closed)" if sm_c is None
                       else f"Safe-mIoU regressed ({sm_c} vs {sm_ch})")
    if not safety_ok:
        reasons.append(f"safety-class regression: {regressed}")
    reasons += rec_floor["reasons"] + rec_reg["reasons"]
    if promote:
        reasons.append("beats champion without any safety regression")
    return {"promote": promote, "beats_map": beats_map, "map_delta": round(map_c - map_ch, 4),
            "safe_ok": safe_ok, "safety_ok": safety_ok, "regressed_safety": regressed,
            "recall_ok": recall_ok, "reasons": reasons}


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
    return chal_m, champ_m, f"common_gold:{gold_id}"


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
        return {"promoted": True, "champion": challenger_version, "promoted_from": prev, "gate": gate}

    await record(db, "champion", "reject", challenger_version, {"gate": gate})
    log.info("govern.reject", challenger=challenger_version, reasons=gate["reasons"])
    return {"promoted": False, "gate": gate, "alert": "challenger rejected: " + "; ".join(gate["reasons"])}
